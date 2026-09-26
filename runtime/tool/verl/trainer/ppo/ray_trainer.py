# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean, masked_whiten



def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                      algorithm_config=None, scppo_lagged_coefficients=None):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

    elif adv_estimator == 'grpo_no_std':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_no_std_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns


    elif adv_estimator in (
            'gdpo', 'dvao', 'scppo_shadow', 'scppo', 'scppo_lagged',
            'scppo_sweep_shadow'):
        if algorithm_config is None:
            raise ValueError(f'{adv_estimator} requires algorithm_config')
        component_names = list(algorithm_config.scppo.component_names)
        reward_components = _configured_reward_components(data, component_names)
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        diagnostics = {}
        if adv_estimator == 'gdpo':
            if len(algorithm_config.gdpo.reward_weights) != len(component_names):
                raise ValueError('GDPO weights must match the shared reward component order')
            advantages, returns = core_algos.compute_gdpo_outcome_advantage(
                reward_components=reward_components,
                eos_mask=response_mask,
                index=index,
                reward_weights=list(algorithm_config.gdpo.reward_weights),
            )
        elif adv_estimator == 'dvao':
            if list(algorithm_config.dvao.component_names) != component_names:
                raise ValueError('DVAO and SC-PPO must use the same reward component order')
            advantages, returns, diagnostics = core_algos.compute_dvao_outcome_advantage(
                reward_components=reward_components,
                eos_mask=response_mask,
                index=index,
                base_weights=list(algorithm_config.dvao.base_weights),
                reward_bounds=[list(bounds) for bounds in algorithm_config.dvao.reward_bounds],
                component_names=list(algorithm_config.dvao.component_names),
                epsilon=float(algorithm_config.dvao.epsilon),
                std_correction=int(algorithm_config.dvao.std_correction),
            )
        elif adv_estimator == 'scppo_lagged':
            if scppo_lagged_coefficients is None:
                raise ValueError('scppo_lagged requires detached coefficients from prior state')
            scppo_config = algorithm_config.scppo
            advantages, returns, diagnostics = core_algos.compute_lagged_scppo_outcome_advantage(
                reward_components=reward_components,
                eos_mask=response_mask,
                index=index,
                coefficients=scppo_lagged_coefficients,
                component_names=list(scppo_config.component_names),
            )
        elif adv_estimator in ('scppo_shadow', 'scppo'):
            scppo_config = algorithm_config.scppo
            advantages, returns, diagnostics = core_algos.compute_scppo_outcome_advantage(
                reward_components=reward_components,
                eos_mask=response_mask,
                index=index,
                component_names=list(scppo_config.component_names),
                targets=list(scppo_config.targets),
                priority_floor=float(scppo_config.priority_floor),
                priority_power=float(scppo_config.priority_power),
                signal_min_group_fraction=float(scppo_config.signal_min_group_fraction),
                signal_min_relative_norm=float(scppo_config.signal_min_relative_norm),
                saturation_tolerance=float(scppo_config.saturation_tolerance),
                crossfit_folds=int(scppo_config.crossfit_folds),
                confidence_z=float(scppo_config.confidence_z),
                ridge=float(scppo_config.ridge),
                gdpo_reward_weights=list(algorithm_config.gdpo.reward_weights),
                dvao_base_weights=list(algorithm_config.dvao.base_weights),
                dvao_reward_bounds=[list(bounds) for bounds in algorithm_config.dvao.reward_bounds],
                dvao_epsilon=float(algorithm_config.dvao.epsilon),
                dvao_std_correction=int(algorithm_config.dvao.std_correction),
                reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
                shadow=adv_estimator == 'scppo_shadow',
            )
        else:
            scppo_config = algorithm_config.scppo
            advantages, returns = core_algos.compute_gdpo_outcome_advantage(
                reward_components=reward_components,
                eos_mask=response_mask,
                index=index,
                reward_weights=list(algorithm_config.gdpo.reward_weights),
            )
            for variant in list(scppo_config.sweep_variants):
                variant_name = str(variant.name)
                crossfit_folds = int(variant.crossfit_folds)
                _, _, variant_metrics = core_algos.compute_scppo_outcome_advantage(
                    reward_components=reward_components,
                    eos_mask=response_mask,
                    index=index,
                    component_names=list(scppo_config.component_names),
                    targets=list(scppo_config.targets),
                    priority_floor=float(scppo_config.priority_floor),
                    priority_power=float(scppo_config.priority_power),
                    signal_min_group_fraction=float(variant.signal_min_group_fraction),
                    signal_min_relative_norm=float(variant.signal_min_relative_norm),
                    saturation_tolerance=float(scppo_config.saturation_tolerance),
                    crossfit_folds=int(crossfit_folds),
                    confidence_z=float(scppo_config.confidence_z),
                    ridge=float(scppo_config.ridge),
                    gdpo_reward_weights=list(algorithm_config.gdpo.reward_weights),
                    dvao_base_weights=list(algorithm_config.dvao.base_weights),
                    dvao_reward_bounds=[list(bounds) for bounds in algorithm_config.dvao.reward_bounds],
                    dvao_epsilon=float(algorithm_config.dvao.epsilon),
                    dvao_std_correction=int(algorithm_config.dvao.std_correction),
                    reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
                    shadow=True,
                )
                for metric_name, metric_value in variant_metrics.items():
                    suffix = metric_name.removeprefix('scppo/')
                    diagnostics[f'scppo_sweep/{variant_name}/{suffix}'] = metric_value
            diagnostics['scppo_sweep/update_is_gdpo'] = 1.0
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        data.meta_info['advantage_metrics'] = diagnostics

    else:
        raise NotImplementedError
    return data


def select_dataproto(data: DataProto, positions) -> DataProto:
    """Return a batch subset while preserving tensor/non-tensor alignment."""
    positions_np = np.asarray(positions, dtype=np.int64)
    positions_torch = torch.as_tensor(positions_np, dtype=torch.long)
    batch = data.batch[positions_torch] if data.batch is not None else None
    non_tensor_batch = {key: value[positions_np] for key, value in data.non_tensor_batch.items()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=dict(data.meta_info))


def _configured_reward_components(data: DataProto, component_names) -> list[torch.Tensor]:
    """Read reward tensors in the single order shared by every algorithm."""
    names = [str(name) for name in component_names]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('Multi-reward component names must be unique and contain at least two entries')
    components = []
    for name in names:
        key = f'token_level_scores_{name}'
        if key not in data.batch:
            raise KeyError(f'Missing configured reward component: {key}')
        components.append(data.batch[key])
    return components


def select_reward_varying_groups(data: DataProto, sequence_rewards: torch.Tensor,
                                 expected_group_size: int, epsilon: float = 1e-6):
    """Keep complete rollout groups whose sequence reward is not constant."""
    if sequence_rewards.ndim != 1 or len(sequence_rewards) != len(data):
        raise ValueError('Dynamic sampling requires one sequence reward per response')
    if 'uid' not in data.non_tensor_batch:
        raise ValueError('Dynamic sampling requires uid group identifiers')

    grouped_positions = {}
    for position, uid in enumerate(data.non_tensor_batch['uid']):
        grouped_positions.setdefault(uid, []).append(position)

    selected_positions = []
    for uid, positions in grouped_positions.items():
        if len(positions) != expected_group_size:
            raise ValueError(
                f'Rollout group {uid} has {len(positions)} responses; expected {expected_group_size}'
            )
        group_rewards = sequence_rewards[torch.as_tensor(positions, dtype=torch.long)]
        if (group_rewards.max() - group_rewards.min()).item() > epsilon:
            selected_positions.extend(positions)

    return select_dataproto(data, selected_positions), len(selected_positions) // expected_group_size, len(grouped_positions)


def select_first_complete_groups(data: DataProto, num_groups: int, expected_group_size: int) -> DataProto:
    """Take exactly the first N uid groups without splitting a rollout group."""
    grouped_positions = {}
    for position, uid in enumerate(data.non_tensor_batch['uid']):
        grouped_positions.setdefault(uid, []).append(position)
    if len(grouped_positions) < num_groups:
        raise ValueError(f'Only {len(grouped_positions)} groups available; need {num_groups}')

    selected_positions = []
    for positions in list(grouped_positions.values())[:num_groups]:
        if len(positions) != expected_group_size:
            raise ValueError('Cannot select an incomplete rollout group')
        selected_positions.extend(positions)
    return select_dataproto(data, selected_positions)


def prepare_scppo_calibration_batch(data: DataProto, scppo_config,
                                    num_groups: int, expected_group_size: int) -> DataProto:
    """Select deterministic complete groups and attach independent objectives."""
    grouped_positions = {}
    for position, uid in enumerate(data.non_tensor_batch['uid']):
        grouped_positions.setdefault(uid, []).append(position)
    if len(grouped_positions) < num_groups:
        raise ValueError(f'Calibration needs {num_groups} groups; found {len(grouped_positions)}')
    ranked = sorted(
        grouped_positions.items(),
        key=lambda item: hashlib.sha256(str(item[0]).encode('utf-8')).digest(),
    )
    selected_positions = []
    selected_uids = []
    for uid, positions in ranked[:num_groups]:
        if len(positions) != expected_group_size:
            raise ValueError(
                f'Calibration group {uid} has {len(positions)} responses; expected {expected_group_size}'
            )
        selected_positions.extend(positions)
        selected_uids.append(str(uid))

    full_response_length = data.batch['responses'].size(-1)
    full_response_mask = data.batch['attention_mask'][:, -full_response_length:]
    _, full_statistics = core_algos.compute_scppo_calibration_inputs(
        reward_components=_configured_reward_components(
            data, list(scppo_config.component_names)
        ),
        eos_mask=full_response_mask,
        index=data.non_tensor_batch['uid'],
        targets=list(scppo_config.targets),
        priority_floor=float(scppo_config.priority_floor),
        priority_power=float(scppo_config.priority_power),
        signal_min_group_fraction=float(scppo_config.signal_min_group_fraction),
        signal_min_relative_norm=float(scppo_config.signal_min_relative_norm),
        saturation_tolerance=float(scppo_config.saturation_tolerance),
        ridge=float(scppo_config.ridge),
        reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
    )
    calibration = select_dataproto(data, selected_positions)
    response_length = calibration.batch['responses'].size(-1)
    response_mask = calibration.batch['attention_mask'][:, -response_length:]
    objective_advantages, statistics = core_algos.compute_scppo_calibration_inputs(
        reward_components=_configured_reward_components(
            calibration, list(scppo_config.component_names)
        ),
        eos_mask=response_mask,
        index=calibration.non_tensor_batch['uid'],
        targets=list(scppo_config.targets),
        priority_floor=float(scppo_config.priority_floor),
        priority_power=float(scppo_config.priority_power),
        signal_min_group_fraction=float(scppo_config.signal_min_group_fraction),
        signal_min_relative_norm=float(scppo_config.signal_min_relative_norm),
        saturation_tolerance=float(scppo_config.saturation_tolerance),
        ridge=float(scppo_config.ridge),
        objective_centering_means=full_statistics['objective_centering_means'],
        reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
    )
    subset_reward_means = statistics['reward_means']
    subset_coverage = statistics['coverage']
    proxy_coefficients, proxy_active, priorities = core_algos.scppo_coefficients_from_gram(
        gram=statistics['proxy_gram'],
        reward_means=full_statistics['reward_means'],
        coverage=full_statistics['coverage'],
        targets=list(scppo_config.targets),
        priority_floor=float(scppo_config.priority_floor),
        priority_power=float(scppo_config.priority_power),
        signal_min_group_fraction=float(scppo_config.signal_min_group_fraction),
        signal_min_relative_norm=float(scppo_config.signal_min_relative_norm),
        saturation_tolerance=float(scppo_config.saturation_tolerance),
        ridge=float(scppo_config.ridge),
        reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
    )
    statistics.update({
        'reward_means': full_statistics['reward_means'],
        'coverage': full_statistics['coverage'],
        'proxy_coefficients': proxy_coefficients.numpy().tolist(),
        'proxy_active': proxy_active.to(torch.int64).numpy().tolist(),
        'priorities': priorities.numpy().tolist(),
        'subset_reward_means': subset_reward_means,
        'subset_coverage': subset_coverage,
    })
    statistics['centering_source_group_count'] = len(grouped_positions)
    calibration.batch['objective_advantages'] = objective_advantages
    calibration.meta_info = {
        'temperature': float(data.meta_info.get('temperature', 1.0)),
        'calibration_statistics': statistics,
        'calibration_uid_digest': hashlib.sha256(
            '\n'.join(selected_uids).encode('utf-8')
        ).hexdigest(),
    }
    return calibration


def attach_moco_cagrad_full_batch(data: DataProto, scppo_config) -> dict:
    """Attach every configured objective advantage without reordering samples."""
    response_length = data.batch['responses'].size(-1)
    response_mask = data.batch['attention_mask'][:, -response_length:]
    component_names = list(scppo_config.component_names)
    objective_advantages, statistics = core_algos.compute_scppo_calibration_inputs(
        reward_components=_configured_reward_components(data, component_names),
        eos_mask=response_mask,
        index=data.non_tensor_batch['uid'],
        targets=list(scppo_config.targets),
        priority_floor=float(scppo_config.priority_floor),
        priority_power=float(scppo_config.priority_power),
        signal_min_group_fraction=float(scppo_config.signal_min_group_fraction),
        signal_min_relative_norm=float(scppo_config.signal_min_relative_norm),
        saturation_tolerance=float(scppo_config.saturation_tolerance),
        ridge=float(scppo_config.ridge),
        reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
    )
    data.batch['moco_objective_advantages'] = objective_advantages
    # BFCL irrelevance examples have no tool call in the reference answer.
    # Carry this inexpensive per-sample mask to the actor so an amortized
    # geometry update can protect the rare negative-tool-use objective.
    reward_models = data.non_tensor_batch.get('reward_model')
    if reward_models is None or len(reward_models) != len(data):
        raise ValueError('MoCo--CAGrad requires aligned reward_model metadata')
    data.batch['moco_rare_mask'] = torch.as_tensor(
        ['<tool_call>' not in str(item.get('ground_truth', '')) for item in reward_models],
        dtype=torch.bool,
        device=objective_advantages.device,
    )
    digest = hashlib.sha256()
    for uid in data.non_tensor_batch['uid']:
        digest.update(str(uid).encode('utf-8'))
        digest.update(b'\n')
    for key in ['responses'] + [f'token_level_scores_{name}' for name in component_names]:
        array = data.batch[key].detach().contiguous().cpu().numpy()
        digest.update(memoryview(array).cast('B'))
    statistics['batch_digest'] = digest.hexdigest()
    return statistics


def summarize_scppo_gradient_calibration(calibration: DataProto, gradient_metrics: dict,
                                         scppo_config) -> dict:
    """Compare proxy/anchor directions in the exact parameter-gradient metric."""
    statistics = calibration.meta_info['calibration_statistics']
    objective_count = len(statistics['reward_means'])

    def matrix(prefix):
        return np.asarray([
            [gradient_metrics[f'{prefix}/{left}/{right}'] for right in range(objective_count)]
            for left in range(objective_count)
        ], dtype=np.float64)

    def solve(gram):
        coefficients, active, priorities = core_algos.scppo_coefficients_from_gram(
            gram=gram,
            reward_means=statistics['reward_means'],
            coverage=statistics['coverage'],
            targets=list(scppo_config.targets),
            priority_floor=float(scppo_config.priority_floor),
            priority_power=float(scppo_config.priority_power),
            signal_min_group_fraction=float(scppo_config.signal_min_group_fraction),
            signal_min_relative_norm=float(scppo_config.signal_min_relative_norm),
            saturation_tolerance=float(scppo_config.saturation_tolerance),
            ridge=float(scppo_config.ridge),
            reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
        )
        return coefficients.numpy(), active.numpy(), priorities.numpy()

    proxy_gram = np.asarray(statistics['proxy_gram'], dtype=np.float64)
    proxy_coefficients = np.asarray(statistics['proxy_coefficients'], dtype=np.float64)
    anchor_gram = matrix('scppo_calibration/anchor_gram')
    anchor_coefficients, anchor_active, _ = solve(anchor_gram)
    output = {
        'scppo_calibration/group_count': float(statistics['group_count']),
        'scppo_calibration/proxy_anchor_coefficient_l1': float(
            np.abs(proxy_coefficients - anchor_coefficients).sum()
        ),
        'scppo_calibration/proxy_anchor_direction_cosine_anchor_metric': (
            core_algos.coefficient_direction_cosine(
                proxy_coefficients, anchor_coefficients, anchor_gram
            )
        ),
    }
    for objective_id in range(objective_count):
        output[f'scppo_calibration/proxy_coefficient/{objective_id}'] = proxy_coefficients[objective_id]
        output[f'scppo_calibration/anchor_coefficient/{objective_id}'] = anchor_coefficients[objective_id]
        output[f'scppo_calibration/anchor_active/{objective_id}'] = float(anchor_active[objective_id])
        output[f'scppo_calibration/anchor_margin/{objective_id}'] = float(
            anchor_gram[objective_id] @ anchor_coefficients
        )

    if gradient_metrics.get('scppo_calibration/mode_exact', 0.0) == 1.0:
        exact_gram = matrix('scppo_calibration/exact_gram')
        exact_coefficients, exact_active, _ = solve(exact_gram)
        anchor_exact_cosine = core_algos.coefficient_direction_cosine(
            anchor_coefficients, exact_coefficients, exact_gram
        )
        proxy_exact_cosine = core_algos.coefficient_direction_cosine(
            proxy_coefficients, exact_coefficients, exact_gram
        )
        output.update({
            'scppo_calibration/exact_available': 1.0,
            'scppo_calibration/anchor_exact_direction_cosine': anchor_exact_cosine,
            'scppo_calibration/proxy_exact_direction_cosine': proxy_exact_cosine,
            'scppo_calibration/anchor_exact_coefficient_l1': float(
                np.abs(anchor_coefficients - exact_coefficients).sum()
            ),
            'scppo_calibration/proxy_exact_coefficient_l1': float(
                np.abs(proxy_coefficients - exact_coefficients).sum()
            ),
            'scppo_calibration/exact_objective_cosine': float(
                exact_gram[0, 1] /
                max(np.sqrt(max(exact_gram[0, 0], 0.0) * max(exact_gram[1, 1], 0.0)), 1e-12)
            ),
        })
        for objective_id in range(objective_count):
            exact_norm = np.sqrt(max(exact_gram[objective_id, objective_id], 0.0))
            anchor_direction_norm = np.sqrt(max(
                anchor_coefficients @ exact_gram @ anchor_coefficients, 0.0
            ))
            raw_margin = float(exact_gram[objective_id] @ anchor_coefficients)
            output[f'scppo_calibration/exact_coefficient/{objective_id}'] = exact_coefficients[objective_id]
            output[f'scppo_calibration/exact_active/{objective_id}'] = float(exact_active[objective_id])
            output[f'scppo_calibration/anchor_exact_margin/{objective_id}'] = raw_margin
            output[f'scppo_calibration/anchor_exact_normalized_margin/{objective_id}'] = (
                raw_margin / max(exact_norm * anchor_direction_norm, 1e-12)
            )
    else:
        output['scppo_calibration/exact_available'] = 0.0
    return output


def resolve_lagged_scppo_coefficients(state, scppo_config, step: int) -> tuple[np.ndarray, int]:
    """Resolve coefficients using only observations strictly before ``step``."""
    objective_count = len(scppo_config.component_names)
    if step < 1 or objective_count < 2:
        raise ValueError('Lagged SC-PPO step and objective count are invalid')
    if step == 1:
        if state is not None:
            raise ValueError('Lagged SC-PPO step 1 must start without prior state')
        return np.full(objective_count, 1.0 / objective_count, dtype=np.float64), 0
    if state is None or int(state['last_step']) != step - 1:
        raise RuntimeError(f'Lagged SC-PPO step {step} lacks state from step {step - 1}')
    coefficients, _, _ = core_algos.scppo_coefficients_from_gram(
        gram=state['ema_anchor_gram'],
        reward_means=state['ema_reward_means'],
        coverage=state['latest_coverage'],
        targets=list(scppo_config.targets),
        priority_floor=float(scppo_config.priority_floor),
        priority_power=float(scppo_config.priority_power),
        signal_min_group_fraction=float(scppo_config.signal_min_group_fraction),
        signal_min_relative_norm=float(scppo_config.signal_min_relative_norm),
        saturation_tolerance=float(scppo_config.saturation_tolerance),
        ridge=float(scppo_config.ridge),
        reward_bounds=[list(bounds) for bounds in scppo_config.reward_bounds],
    )
    return coefficients.numpy(), int(state['last_step'])


def update_lagged_scppo_state(state, anchor_gram, calibration_statistics,
                              step: int, ema_beta: float) -> dict:
    """Commit the current observation for use beginning at the next step."""
    if not 0 <= ema_beta < 1:
        raise ValueError('Lagged SC-PPO EMA beta must lie in [0, 1)')
    current_gram = np.asarray(anchor_gram, dtype=np.float64)
    current_means = np.asarray(calibration_statistics['reward_means'], dtype=np.float64)
    current_coverage = np.asarray(calibration_statistics['coverage'], dtype=np.float64)
    objective_count = len(current_means)
    if current_gram.shape != (objective_count, objective_count):
        raise ValueError('Lagged SC-PPO state Gram shape does not match rewards')
    if not all(np.isfinite(value).all() for value in (
            current_gram, current_means, current_coverage)):
        raise ValueError('Lagged SC-PPO state observation must be finite')
    if state is None:
        if step != 1:
            raise RuntimeError('Lagged SC-PPO state can only initialize at step 1')
        ema_gram = current_gram
        ema_means = current_means
    else:
        if int(state['last_step']) != step - 1:
            raise RuntimeError('Lagged SC-PPO state updates must be consecutive')
        ema_gram = ema_beta * np.asarray(state['ema_anchor_gram']) + (1 - ema_beta) * current_gram
        ema_means = ema_beta * np.asarray(state['ema_reward_means']) + (1 - ema_beta) * current_means
    return {
        'last_step': int(step),
        'ema_anchor_gram': ema_gram.tolist(),
        'ema_reward_means': ema_means.tolist(),
        'latest_coverage': current_coverage.tolist(),
    }


def evaluate_lagged_scppo_gate(calibration: DataProto, gradient_metrics: dict,
                               coefficients, step: int, scppo_config) -> dict:
    """Evaluate the current anchor and prompt-bootstrap safety conditions."""
    statistics = calibration.meta_info['calibration_statistics']
    objective_count = len(statistics['reward_means'])
    weights = np.asarray(coefficients, dtype=np.float64)
    if weights.shape != (objective_count,) or np.any(weights < 0) or not np.isclose(
            weights.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError('Treatment coefficients must be non-negative and sum to one')
    anchor_gram = np.asarray([
        [gradient_metrics[f'scppo_calibration/anchor_gram/{left}/{right}']
         for right in range(objective_count)]
        for left in range(objective_count)
    ], dtype=np.float64)
    anchor_margins = anchor_gram @ weights
    direction_norm = np.sqrt(max(float(weights @ anchor_gram @ weights), 0.0))
    normalized_anchor_margins = [
        float(anchor_margins[objective_id]) / max(
            np.sqrt(max(anchor_gram[objective_id, objective_id], 0.0)) * direction_norm,
            1e-12,
        )
        for objective_id in range(objective_count)
    ]
    response_length = calibration.batch['responses'].size(-1)
    response_mask = calibration.batch['attention_mask'][:, -response_length:]
    proxy_means, proxy_lcbs = core_algos.prompt_bootstrap_directional_lcbs(
        objective_advantages=calibration.batch['objective_advantages'],
        response_mask=response_mask,
        index=calibration.non_tensor_batch['uid'],
        coefficients=weights,
        replicates=int(scppo_config.treatment.bootstrap_replicates),
        seed=int(scppo_config.treatment.bootstrap_seed_base) + int(step),
        quantile=float(scppo_config.treatment.bootstrap_lower_quantile),
    )
    natural_accept = bool(np.all(anchor_margins >= 0.0) and np.all(np.asarray(proxy_lcbs) >= 0.0))
    forced_decision = scppo_config.treatment.get('force_gate_decision')
    if forced_decision not in (None, 'reject'):
        raise ValueError('force_gate_decision is restricted to null or reject')
    accepted = natural_accept and forced_decision is None
    result = {
        'accepted': accepted,
        'natural_accept': natural_accept,
        'forced_reject': forced_decision == 'reject',
        'coefficients': weights.tolist(),
        'anchor_gram': anchor_gram.tolist(),
        'anchor_margins': anchor_margins.tolist(),
        'normalized_anchor_margins': normalized_anchor_margins,
        'proxy_margin_means': proxy_means,
        'proxy_lcb95': proxy_lcbs,
        'rejection_reason': (
            'forced_preflight_reject' if forced_decision == 'reject' else
            'anchor_margin' if np.any(anchor_margins < 0.0) else
            'bootstrap_lcb' if np.any(np.asarray(proxy_lcbs) < 0.0) else
            'none'
        ),
    }
    if gradient_metrics.get('scppo_calibration/mode_exact', 0.0) == 1.0:
        exact_gram = np.asarray([
            [gradient_metrics[f'scppo_calibration/exact_gram/{left}/{right}']
             for right in range(objective_count)]
            for left in range(objective_count)
        ], dtype=np.float64)
        exact_margins = exact_gram @ weights
        exact_direction_norm = np.sqrt(max(float(weights @ exact_gram @ weights), 0.0))
        result['exact_gram'] = exact_gram.tolist()
        result['exact_margins'] = exact_margins.tolist()
        result['normalized_exact_margins'] = [
            float(exact_margins[objective_id]) / max(
                np.sqrt(max(exact_gram[objective_id, objective_id], 0.0)) * exact_direction_norm,
                1e-12,
            )
            for objective_id in range(objective_count)
        ]
    return result


def resolve_generation_group_batch_size(target_groups: int, dynamic_sampling: bool,
                                        configured_size) -> int:
    """Resolve an execution-only rollout batch without changing the PPO batch."""
    if not dynamic_sampling or configured_size is None:
        return target_groups
    generation_groups = int(configured_size)
    if generation_groups < target_groups or generation_groups % target_groups != 0:
        raise ValueError(
            'filter_groups.generation_batch_size must be an integer multiple of '
            f'data.train_batch_size ({target_groups}), got {generation_groups}'
        )
    return generation_groups


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    
    sequence_score_format = batch.batch['token_level_scores_format'].sum(-1)
    sequence_score_correctness = batch.batch['token_level_scores_correctness'].sum(-1)
    sequence_score_length = batch.batch['token_level_scores_length'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # format score
        'critic/format_score/mean':
            torch.mean(sequence_score_format).detach().item(),
        'critic/format_score/max':
            torch.max(sequence_score_format).detach().item(),
        'critic/format_score/min':
            torch.min(sequence_score_format).detach().item(),
        # correctness score
        'critic/correctness_score/mean':
            torch.mean(sequence_score_correctness).detach().item(),
        'critic/correctness_score/max':
            torch.max(sequence_score_correctness).detach().item(),
        'critic/correctness_score/min':
            torch.min(sequence_score_correctness).detach().item(),
        # length score
        'critic/length_score/mean':
            torch.mean(sequence_score_length).detach().item(),
        'critic/length_score/max':
            torch.max(sequence_score_length).detach().item(),
        'critic/length_score/min':
            torch.min(sequence_score_length).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/var':
            torch.var(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/tokens':
            torch.sum(response_length).detach().item(),
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/tokens':
            torch.sum(prompt_length).detach().item(),
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.scppo_lagged_state = None

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         # use_chat_template=self.config.data.use_chat_template,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='left')
        train_generator = torch.Generator().manual_seed(int(self.config.trainer.get('seed', 1)))
        self._train_generator = train_generator
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           generator=train_generator)

        self.val_dataset = None
        self.val_dataloader = None
        if self.config.data.get('val_files'):
            val_generator = torch.Generator().manual_seed(int(self.config.trainer.get('seed', 1)) + 1)
            self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                           tokenizer=self.tokenizer,
                                           prompt_key=self.config.data.prompt_key,
                                           max_prompt_length=self.config.data.max_prompt_length,
                                           filter_prompts=True,
                                           return_raw_chat=self.config.data.get('return_raw_chat', False),
                                           truncation='left')
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                             batch_size=len(self.val_dataset),
                                             shuffle=True,
                                             drop_last=True,
                                             collate_fn=collate_fn,
                                             generator=val_generator)

        assert len(self.train_dataloader) >= 1
        if self.val_dataloader is not None:
            assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader) if self.val_dataloader is not None else 0}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        if self.val_dataloader is None:
            raise RuntimeError('Validation requested without data.val_files')
        reward_tensor_lst = []
        format_tensor_lst = []
        correctness_tensor_lst = []
        length_tensor_lst = []
        
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor, format_tensor, correctness_tensor, length_tensor = self.val_reward_fn(test_batch, self.global_steps)

            reward_tensor_lst.append(reward_tensor)
            format_tensor_lst.append(format_tensor)
            correctness_tensor_lst.append(correctness_tensor)
            length_tensor_lst.append(length_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        format_tensor = torch.cat(format_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        correctness_tensor = torch.cat(correctness_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        length_tensor = torch.cat(length_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_format = {}
        data_source_correctness = {}
        data_source_length = {}
        
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_format[data_source] = []
                data_source_correctness[data_source] = []
                data_source_length[data_source] = []
            
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_format[data_source].append(format_tensor[i].item())
            data_source_correctness[data_source].append(correctness_tensor[i].item())
            data_source_length[data_source].append(length_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            metric_dict[f'val/test_format/{data_source}'] = np.mean(data_source_format[data_source])
            metric_dict[f'val/test_correctness/{data_source}'] = np.mean(data_source_correctness[data_source])
            metric_dict[f'val/test_length/{data_source}'] = np.mean(data_source_length[data_source])

        return metric_dict

    
    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.adv_estimator == 'grpo_no_std':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in (
                'gdpo', 'dvao', 'scppo_shadow', 'scppo', 'scppo_lagged',
                'scppo_sweep_shadow'):
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        if self.config.trainer.get('complete_checkpoint', False):
            from verl.utils.complete_checkpoint import capture_rng, commit_snapshot, config_contract
            if self.use_critic or self.config.algorithm.adv_estimator not in ('gdpo', 'dvao'):
                raise ValueError('Complete checkpoints currently support GDPO/DVAO only')
            root = self.config.trainer.default_local_dir
            os.makedirs(root, exist_ok=True)
            pending = os.path.join(root, f'.saving_{self.global_steps}_{uuid.uuid4().hex}')
            self.actor_rollout_wg.save_checkpoint(pending, None, True)
            torch.save({
                'global_step': self.global_steps,
                'rng': capture_rng(cuda=False),
                'epoch': self._consumed_data_epochs,
                'batches_in_epoch': self._batches_in_epoch,
                'epoch_generator_state': self._epoch_generator_state,
                'generator_state': self._train_generator.get_state(),
                'kl_ctrl': vars(self.kl_ctrl).copy(),
                'config_contract': config_contract(self.config),
                'runtime_sha256': os.environ.get('TRAIN_RUNTIME_SHA256'),
                'dataset_sha256': os.environ.get('TRAIN_DATASET_SHA256'),
            }, os.path.join(pending, 'trainer_state.pt'))
            commit_snapshot(root, pending, self.global_steps, self.actor_rollout_wg.world_size)
            print(f'COMPLETE_CHECKPOINT step={self.global_steps} path={root}/latest', flush=True)
            return
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None # if self.config.trainer.default_hdfs_dir is None else os.path.join(
            # self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None # if self.config.trainer.default_hdfs_dir is None else os.path.join(
                # self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

        keep_latest = int(self.config.trainer.get('checkpoint_keep_latest', 0))
        if keep_latest > 0:
            for component in ('actor', 'critic'):
                component_dir = os.path.dirname(
                    os.path.join(self.config.trainer.default_local_dir,
                                 component, f'global_step_{self.global_steps}')
                )
                if not os.path.isdir(component_dir):
                    continue
                candidates = []
                for name in os.listdir(component_dir):
                    if not name.startswith('global_step_'):
                        continue
                    try:
                        step = int(name.rsplit('_', 1)[1])
                    except ValueError:
                        continue
                    path = os.path.join(component_dir, name)
                    if os.path.isdir(path):
                        candidates.append((step, path))
                candidates.sort(reverse=True)
                for _, path in candidates[keep_latest:]:
                    shutil.rmtree(path, ignore_errors=True)
            with open(os.path.join(self.config.trainer.default_local_dir,
                                   'latest_checkpoint.json'), 'w', encoding='utf-8') as metadata_file:
                json.dump({'global_step': int(self.global_steps)}, metadata_file)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _next_train_batch(self):
        try:
            batch = next(self._train_dataloader_iter)
        except StopIteration:
            self._epoch_generator_state = self._train_generator.get_state()
            self._train_dataloader_iter = iter(self.train_dataloader)
            self._consumed_data_epochs += 1
            self._batches_in_epoch = 0
            batch = next(self._train_dataloader_iter)
        self._batches_in_epoch += 1
        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        resume_step = int(self.config.trainer.get('resume_step', 0))
        if resume_step < 0 or resume_step > self.total_training_steps:
            raise ValueError(f'resume_step must be in [0, {self.total_training_steps}], got {resume_step}')
        self.global_steps = resume_step

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # Resume from the first batch after the last committed step. The
        # dataloader is deterministic for a fixed seed; advancing to the same
        # offset preserves the epoch/step alignment across a process restart.
        self.global_steps += 1
        self._epoch_generator_state = self._train_generator.get_state()
        self._train_dataloader_iter = iter(self.train_dataloader)
        self._consumed_data_epochs = 0
        self._batches_in_epoch = 0
        if resume_step and self.config.trainer.get('complete_checkpoint', False):
            from verl.utils.complete_checkpoint import restore_rng, config_contract, validate_snapshot
            resume_path = self.config.actor_rollout_ref.actor.resume_from_checkpoint
            manifest = validate_snapshot(resume_path)
            saved = torch.load(os.path.join(resume_path, 'trainer_state.pt'),
                               map_location='cpu', weights_only=False)
            if manifest['global_step'] != resume_step or saved['global_step'] != resume_step:
                raise ValueError('Resume step disagrees with committed checkpoint')
            if saved['config_contract'] != config_contract(self.config):
                raise ValueError('Training configuration changed during resume')
            for field, env in [('runtime_sha256', 'TRAIN_RUNTIME_SHA256'),
                               ('dataset_sha256', 'TRAIN_DATASET_SHA256')]:
                if not saved[field] or saved[field] != os.environ.get(env):
                    raise ValueError(f'Resume provenance mismatch: {field}')
            self._epoch_generator_state = saved['epoch_generator_state']
            self._train_generator.set_state(self._epoch_generator_state)
            self._train_dataloader_iter = iter(self.train_dataloader)
            self._consumed_data_epochs = saved['epoch']
            for _ in range(saved['batches_in_epoch']):
                next(self._train_dataloader_iter)
            self._batches_in_epoch = saved['batches_in_epoch']
            if not torch.equal(self._train_generator.get_state(), saved['generator_state']):
                raise ValueError('DataLoader generator replay mismatch')
            vars(self.kl_ctrl).update(saved['kl_ctrl'])
            restore_rng(saved['rng'])
            print(f'COMPLETE_DRIVER_RESTORE step={resume_step} epoch={saved["epoch"]} '
                  f'consumed_batches={self._batches_in_epoch}', flush=True)
        else:
            for _ in range(resume_step):
                self._next_train_batch()

        while self.global_steps <= self.total_training_steps:
                print(f'data epoch {self._consumed_data_epochs}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}
                actor_update_allowed = True
                treatment_coefficients = None
                treatment_state_source_step = None
                rejected_step_state_before = None

                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                        target_groups = int(self.config.data.train_batch_size)
                        filter_config = self.config.algorithm.filter_groups
                        dynamic_sampling = bool(filter_config.enable)
                        if dynamic_sampling and filter_config.metric != 'seq_reward':
                            raise ValueError('This math reproduction supports dynamic metric=seq_reward only')
                        max_gen_batches = int(filter_config.max_num_gen_batches)
                        generation_group_batch_size = resolve_generation_group_batch_size(
                            target_groups,
                            dynamic_sampling,
                            filter_config.get('generation_batch_size'),
                        )
                        source_batches_per_generation = generation_group_batch_size // target_groups
                        accepted_batches = []
                        accepted_groups = 0
                        generated_groups = 0
                        generation_batches = 0

                        while accepted_groups < target_groups:
                            source_batches = [
                                DataProto.from_single_dict(self._next_train_batch())
                                for _ in range(source_batches_per_generation)
                            ]
                            candidate = source_batches[0] if len(source_batches) == 1 \
                                else DataProto.concat(source_batches)
                            candidate.non_tensor_batch['uid'] = np.array([
                                f'{self.global_steps}:{generation_batches}:{position}'
                                for position in range(len(candidate))
                            ], dtype=object)
                            gen_batch = candidate.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                            gen_batch.meta_info = {
                                'recompute_log_prob': False,
                                'do_sample': True,
                            }
                            generated = self.actor_rollout_wg.generate_sequences(gen_batch)
                            candidate = candidate.repeat(repeat_times=rollout_n, interleave=True)
                            candidate = candidate.union(generated)

                            if self.use_rm:
                                rm_scores = self.rm_wg.compute_rm_score(candidate)
                                candidate = candidate.union(rm_scores)
                            reward_tensor, format_tensor, correctness_tensor, length_tensor = self.reward_fn(
                                candidate, self.global_steps
                            )
                            candidate.batch['token_level_scores'] = reward_tensor
                            candidate.batch['token_level_scores_format'] = format_tensor
                            candidate.batch['token_level_scores_correctness'] = correctness_tensor
                            candidate.batch['token_level_scores_length'] = length_tensor
                            candidate.batch['token_level_rewards'] = reward_tensor

                            sequence_rewards = reward_tensor.sum(dim=-1).cpu()
                            generation_batches += 1
                            generated_groups += len(candidate) // rollout_n
                            if dynamic_sampling:
                                candidate, kept_groups, _ = select_reward_varying_groups(
                                    candidate,
                                    sequence_rewards=sequence_rewards,
                                    expected_group_size=rollout_n,
                                )
                            else:
                                kept_groups = len(candidate) // rollout_n
                            if kept_groups:
                                accepted_batches.append(candidate)
                                accepted_groups += kept_groups

                            if accepted_groups < target_groups and max_gen_batches > 0 and \
                                    generation_batches >= max_gen_batches:
                                raise RuntimeError(
                                    f'Dynamic sampling kept only {accepted_groups}/{target_groups} groups '
                                    f'after {generation_batches} generation batches'
                                )

                        batch = DataProto.concat(accepted_batches)
                        batch = select_first_complete_groups(
                            batch,
                            num_groups=target_groups,
                            expected_group_size=rollout_n,
                        )
                        metrics.update({
                            'dynamic/generated_prompt_groups': generated_groups,
                            'dynamic/generation_prompt_batch_size': generation_group_batch_size,
                            'dynamic/kept_prompt_groups_before_trim': accepted_groups,
                            'dynamic/generation_batches': generation_batches,
                            'dynamic/keep_fraction': accepted_groups / max(generated_groups, 1),
                        })

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_actor_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        if self.config.algorithm.adv_estimator == 'scppo_lagged':
                            treatment_coefficients, treatment_state_source_step = (
                                resolve_lagged_scppo_coefficients(
                                    state=self.scppo_lagged_state,
                                    scppo_config=self.config.algorithm.scppo,
                                    step=self.global_steps,
                                )
                            )

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  algorithm_config=self.config.algorithm,
                                                  scppo_lagged_coefficients=treatment_coefficients)
                        metrics.update(batch.meta_info.get('advantage_metrics', {}))
                        moco_config = self.config.actor_rollout_ref.actor.get(
                            'moco_cagrad_shadow', {}
                        )
                        if bool(moco_config.get('enabled', False)):
                            if self.config.algorithm.adv_estimator != 'scppo_shadow':
                                raise ValueError(
                                    'MoCo--CAGrad requires the per-objective '
                                    'scppo_shadow advantage path'
                                )
                            moco_statistics = attach_moco_cagrad_full_batch(
                                batch, self.config.algorithm.scppo
                            )
                            expected_objectives = int(moco_config.objective_count)
                            if batch.batch['moco_objective_advantages'].shape[1] != expected_objectives:
                                raise ValueError('MoCo--CAGrad objective count does not match config')
                            batch.meta_info.update({
                                'moco_outer_step': int(self.global_steps),
                                'moco_uid_digest': moco_statistics['batch_digest'],
                            })
                            moco_prefix = (
                                'moco_active'
                                if bool(moco_config.get('apply_update', False))
                                else 'moco_shadow'
                            )
                            metrics.update({
                                f'{moco_prefix}/objective_count': float(expected_objectives),
                                f'{moco_prefix}/group_count': float(moco_statistics['group_count']),
                            })
                        if (self.config.algorithm.adv_estimator == 'scppo_lagged' and
                                self.global_steps == 1 and
                                bool(self.config.algorithm.scppo.treatment.get(
                                    'assert_step1_gdpo_equivalence', True))):
                            response_length = batch.batch['responses'].size(-1)
                            response_mask = batch.batch['attention_mask'][:, -response_length:]
                            gdpo_advantages, _ = core_algos.compute_gdpo_outcome_advantage(
                                reward_components=_configured_reward_components(
                                    batch,
                                    list(self.config.algorithm.scppo.component_names),
                                ),
                                eos_mask=response_mask,
                                index=batch.non_tensor_batch['uid'],
                                reward_weights=list(self.config.algorithm.gdpo.reward_weights),
                            )
                            if not torch.equal(batch.batch['advantages'], gdpo_advantages):
                                maximum_error = (
                                    batch.batch['advantages'] - gdpo_advantages
                                ).abs().max().item()
                                raise RuntimeError(
                                    'Lagged SC-PPO uniform prior is not bitwise GDPO-equivalent; '
                                    f'max_abs_error={maximum_error}'
                                )
                            metrics['scppo_treatment/step1_bitwise_gdpo_equivalent'] = 1.0

                    calibration_config = self.config.algorithm.scppo.get('calibration', {})
                    calibration_enabled = bool(calibration_config.get('enabled', False))
                    if (self.config.algorithm.adv_estimator == 'scppo_lagged' and
                            not calibration_enabled):
                        raise ValueError('scppo_lagged requires current-step anchor calibration')
                    if calibration_enabled:
                        if self.config.algorithm.adv_estimator not in (
                                'scppo_shadow', 'scppo_lagged'):
                            raise ValueError(
                                'Gradient calibration is restricted to shadow or lagged SC-PPO'
                            )
                        calibration_batch = prepare_scppo_calibration_batch(
                            data=batch,
                            scppo_config=self.config.algorithm.scppo,
                            num_groups=int(calibration_config.group_count),
                            expected_group_size=int(self.config.actor_rollout_ref.rollout.n),
                        )
                        self._balance_batch(
                            calibration_batch,
                            metrics=metrics,
                            logging_prefix='calibration_seqlen',
                        )
                        forced_mode = calibration_config.get('force_mode')
                        if forced_mode not in (None, 'anchor', 'exact'):
                            raise ValueError('calibration.force_mode must be null, anchor, or exact')
                        registered_exact_steps = calibration_config.get('exact_steps')
                        if forced_mode is not None:
                            exact_step = forced_mode == 'exact'
                        elif registered_exact_steps is not None:
                            registered_exact_steps = [
                                int(value) for value in registered_exact_steps
                            ]
                            if (not registered_exact_steps or
                                    len(set(registered_exact_steps)) != len(registered_exact_steps) or
                                    min(registered_exact_steps) < 1):
                                raise ValueError('calibration.exact_steps must be unique positive steps')
                            exact_step = self.global_steps in registered_exact_steps
                        else:
                            exact_every = int(calibration_config.exact_every_n_steps)
                            if exact_every < 1:
                                raise ValueError('exact_every_n_steps must be positive')
                            exact_step = (self.global_steps - 1) % exact_every == 0
                        calibration_batch.meta_info.update({
                            'temperature': float(self.config.actor_rollout_ref.rollout.temperature),
                            'gradient_mode': 'exact' if exact_step else 'anchor',
                            'anchor_last_n_layers': int(calibration_config.anchor_last_n_layers),
                        })
                        audit_rejected_state = bool(
                            self.config.algorithm.adv_estimator == 'scppo_lagged' and
                            self.config.algorithm.scppo.treatment.get(
                                'audit_rejected_step_state', False
                            )
                        )
                        if audit_rejected_state:
                            if self.config.algorithm.scppo.treatment.get(
                                    'force_gate_decision') != 'reject':
                                raise ValueError(
                                    'Rejected-step state audit requires force_gate_decision=reject'
                                )
                            rejected_step_state_before = (
                                self.actor_rollout_wg.audit_actor_training_state()
                            )
                        print(
                            f'scppo calibration step={self.global_steps} '
                            f'mode={calibration_batch.meta_info["gradient_mode"]} '
                            f'groups={calibration_config.group_count} '
                            f'uid_digest={calibration_batch.meta_info["calibration_uid_digest"]}',
                            flush=True,
                        )
                        with _timer('scppo_calibration', timing_raw):
                            calibration_output = self.actor_rollout_wg.compute_actor_gradient_gram(
                                calibration_batch
                            )
                        calibration_metrics = reduce_metrics(
                            calibration_output.meta_info['metrics']
                        )
                        metrics.update(calibration_metrics)
                        calibration_summary = summarize_scppo_gradient_calibration(
                            calibration=calibration_batch,
                            gradient_metrics=calibration_metrics,
                            scppo_config=self.config.algorithm.scppo,
                        )
                        metrics.update(calibration_summary)
                        calibration_record = {
                            'step': self.global_steps,
                            'uid_digest': calibration_batch.meta_info['calibration_uid_digest'],
                            **calibration_metrics,
                            **calibration_summary,
                        }
                        calibration_record = {
                            key: value.item() if isinstance(value, np.generic) else value
                            for key, value in calibration_record.items()
                        }
                        print(
                            'scppo_calibration_json=' + json.dumps(
                                calibration_record, sort_keys=True, allow_nan=False
                            ),
                            flush=True,
                        )

                        if self.config.algorithm.adv_estimator == 'scppo_lagged':
                            treatment_gate = evaluate_lagged_scppo_gate(
                                calibration=calibration_batch,
                                gradient_metrics=calibration_metrics,
                                coefficients=treatment_coefficients,
                                step=self.global_steps,
                                scppo_config=self.config.algorithm.scppo,
                            )
                            actor_update_allowed = bool(treatment_gate['accepted'])
                            self.scppo_lagged_state = update_lagged_scppo_state(
                                state=self.scppo_lagged_state,
                                anchor_gram=treatment_gate['anchor_gram'],
                                calibration_statistics=(
                                    calibration_batch.meta_info['calibration_statistics']
                                ),
                                step=self.global_steps,
                                ema_beta=float(
                                    self.config.algorithm.scppo.treatment.ema_beta
                                ),
                            )
                            metrics.update({
                                'scppo_treatment/gate_accepted': float(actor_update_allowed),
                                'scppo_treatment/gate_natural_accept': float(
                                    treatment_gate['natural_accept']
                                ),
                                'scppo_treatment/gate_forced_reject': float(
                                    treatment_gate['forced_reject']
                                ),
                                'scppo_treatment/state_source_step': float(
                                    treatment_state_source_step
                                ),
                                'scppo_treatment/state_committed_step': float(
                                    self.scppo_lagged_state['last_step']
                                ),
                            })
                            for objective_id in range(len(treatment_coefficients)):
                                metrics.update({
                                    f'scppo_treatment/anchor_margin/{objective_id}': (
                                        treatment_gate['anchor_margins'][objective_id]
                                    ),
                                    f'scppo_treatment/anchor_normalized_margin/{objective_id}': (
                                        treatment_gate['normalized_anchor_margins'][objective_id]
                                    ),
                                    f'scppo_treatment/proxy_margin_mean/{objective_id}': (
                                        treatment_gate['proxy_margin_means'][objective_id]
                                    ),
                                    f'scppo_treatment/proxy_lcb95/{objective_id}': (
                                        treatment_gate['proxy_lcb95'][objective_id]
                                    ),
                                })
                                if 'normalized_exact_margins' in treatment_gate:
                                    metrics[
                                        f'scppo_treatment/exact_normalized_margin/{objective_id}'
                                    ] = treatment_gate['normalized_exact_margins'][objective_id]
                            treatment_record = {
                                'step': self.global_steps,
                                'uid_digest': calibration_batch.meta_info[
                                    'calibration_uid_digest'
                                ],
                                'state_source_step': treatment_state_source_step,
                                'committed_state': self.scppo_lagged_state,
                                **treatment_gate,
                            }
                            print(
                                'scppo_treatment_json=' + json.dumps(
                                    treatment_record, sort_keys=True, allow_nan=False
                                ),
                                flush=True,
                            )
                            if audit_rejected_state:
                                if actor_update_allowed:
                                    raise RuntimeError('Forced rejected-step audit unexpectedly accepted')
                                rejected_step_state_after = (
                                    self.actor_rollout_wg.audit_actor_training_state()
                                )
                                if rejected_step_state_after != rejected_step_state_before:
                                    raise RuntimeError(
                                        'Rejected SC-PPO step changed parameter, optimizer, or scheduler state'
                                    )
                                metrics['scppo_treatment/rejected_state_unchanged'] = 1.0

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if (self.config.trainer.critic_warmup <= self.global_steps and
                            actor_update_allowed):
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    if self.config.algorithm.adv_estimator == 'scppo_lagged':
                        metrics['scppo_treatment/actor_update_applied'] = float(
                            actor_update_allowed and
                            self.config.trainer.critic_warmup <= self.global_steps
                        )
                        metrics['scppo_treatment/scheduler_update_applied'] = metrics[
                            'scppo_treatment/actor_update_applied'
                        ]

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                            self.global_steps % self.config.trainer.save_freq == 0 or
                            (self.config.trainer.get('complete_checkpoint', False) and
                             self.global_steps in (1, self.total_training_steps,
                                 int(self.config.trainer.get('stop_after_step', -1))))):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.global_steps == int(self.config.trainer.get('stop_after_step', -1)):
                    print(f'CONTROLLED_STOP step={self.global_steps}', flush=True)
                    return

                self.global_steps += 1

                if self.global_steps > self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
