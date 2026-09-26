# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
from itertools import combinations
from typing import Sequence

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        scores = token_level_rewards.to(torch.float32).sum(dim=-1)
        sequence_advantages = torch.zeros_like(scores)
        for positions in _group_positions(index).values():
            position_tensor = torch.as_tensor(positions, dtype=torch.long, device=scores.device)
            group_scores = scores.index_select(0, position_tensor)
            if len(positions) == 1:
                group_advantages = torch.zeros_like(group_scores)
            else:
                group_advantages = (group_scores - group_scores.mean()) / (
                    group_scores.std(correction=1) + epsilon
                )
            sequence_advantages.index_copy_(0, position_tensor, group_advantages)
        advantages = sequence_advantages.unsqueeze(-1).expand_as(eos_mask) * eos_mask

    return advantages, advantages


def _group_positions(index) -> dict:
    groups = defaultdict(list)
    for position, group_id in enumerate(index):
        try:
            hash(group_id)
            key = group_id
        except TypeError:
            key = str(group_id)
        groups[key].append(position)
    return groups


def compute_gdpo_outcome_advantage(
        reward_components: Sequence[torch.Tensor],
        eos_mask: torch.Tensor,
        index,
        reward_weights: Sequence[float] = None,
        epsilon: float = 1e-6):
    """Official GDPO: normalize each reward per group, sum, then batch-whiten."""
    if len(reward_components) < 2:
        raise ValueError('GDPO requires at least two reward components')
    reference = reward_components[0]
    if any(component.shape != reference.shape for component in reward_components):
        raise ValueError('All GDPO reward components must have identical shapes')
    if eos_mask.shape != reference.shape:
        raise ValueError('GDPO response mask must match reward component shapes')

    if reward_weights is None:
        reward_weights = [1.0] * len(reward_components)
    if len(reward_weights) != len(reward_components):
        raise ValueError('GDPO reward weights must match reward components')
    weights = torch.as_tensor(reward_weights, dtype=torch.float32, device=reference.device)
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('GDPO reward weights must be finite, non-negative, and nonzero')

    with torch.no_grad():
        normalized = []
        for component in reward_components:
            advantage, _ = compute_grpo_outcome_advantage(
                token_level_rewards=component,
                eos_mask=eos_mask,
                index=index,
                epsilon=epsilon,
            )
            normalized.append(advantage)
        combined = torch.stack(normalized, dim=0)
        combined = (combined * weights[:, None, None]).sum(dim=0)
        advantages = verl_F.masked_whiten(combined, eos_mask) * eos_mask
    return advantages, advantages


def _masked_inner_product(left: torch.Tensor, right: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """Token-weighted inner product used only as a score-function proxy."""
    mask = mask.to(torch.float32)
    denominator = mask.sum().clamp_min(1.0)
    return (left.to(torch.float32) * right.to(torch.float32) * mask).sum() / denominator


def _simplex_min_norm_coefficients(gram: torch.Tensor, priorities: torch.Tensor,
                                   active: torch.Tensor, ridge: float = 1e-8) -> torch.Tensor:
    """Solve the priority-scaled MGDA problem for a small objective set.

    The returned coefficients multiply the unscaled objective gradients.  This
    routine is deterministic and exact for the two-objective case used here.
    """
    objective_count = gram.shape[0]
    active_ids = torch.nonzero(active, as_tuple=False).flatten()
    coefficients = torch.zeros(objective_count, dtype=torch.float64, device=gram.device)
    if active_ids.numel() == 0:
        return coefficients
    if active_ids.numel() == 1:
        coefficients[active_ids[0]] = 1.0
        return coefficients

    safe_priorities = priorities.to(torch.float64).clamp_min(ridge)
    scaled = gram.to(torch.float64) / (
        safe_priorities[:, None] * safe_priorities[None, :]
    )
    active_gram = scaled.index_select(0, active_ids).index_select(1, active_ids)
    if active_ids.numel() == 2:
        g11, g12 = active_gram[0, 0], active_gram[0, 1]
        g22 = active_gram[1, 1]
        denominator = g11 + g22 - 2.0 * g12
        if denominator.abs().item() <= ridge:
            alpha_first = active_gram.new_tensor(0.5)
        else:
            alpha_first = ((g22 - g12) / denominator).clamp(0.0, 1.0)
        alpha = torch.stack((alpha_first, 1.0 - alpha_first))
    elif active_ids.numel() <= 10:
        # An optimum of the convex simplex QP is stationary on one of its
        # supports.  Enumerating supports is exact and cheap for reward counts
        # seen in multi-reward RL (2^10 - 1 candidates at most).
        best_alpha = None
        best_value = None
        dimension = active_ids.numel()
        for support_size in range(1, dimension + 1):
            for support_tuple in combinations(range(dimension), support_size):
                support = torch.as_tensor(
                    support_tuple, dtype=torch.long, device=gram.device
                )
                support_gram = active_gram.index_select(0, support).index_select(1, support)
                ones = torch.ones(support_size, dtype=torch.float64, device=gram.device)
                kkt = torch.zeros(
                    (support_size + 1, support_size + 1),
                    dtype=torch.float64, device=gram.device,
                )
                kkt[:support_size, :support_size] = support_gram
                kkt[:support_size, support_size] = -ones
                kkt[support_size, :support_size] = ones
                rhs = torch.zeros(support_size + 1, dtype=torch.float64, device=gram.device)
                rhs[support_size] = 1.0
                solution = torch.linalg.lstsq(kkt, rhs).solution
                support_alpha = solution[:support_size]
                residual = (kkt @ solution - rhs).abs().max()
                if residual.item() > 1e-7 or support_alpha.min().item() < -1e-9:
                    continue
                candidate = torch.zeros(dimension, dtype=torch.float64, device=gram.device)
                candidate[support] = support_alpha.clamp_min(0.0)
                candidate /= candidate.sum().clamp_min(ridge)
                value = candidate @ active_gram @ candidate
                if best_value is None or value.item() < best_value.item():
                    best_alpha, best_value = candidate, value
        if best_alpha is None:
            raise RuntimeError('SC-PPO simplex QP found no feasible active support')
        alpha = best_alpha
    else:
        raise ValueError('SC-PPO supports at most 10 simultaneously active objectives')

    coefficients[active_ids] = alpha / safe_priorities[active_ids]
    return coefficients


def _signal_conflict_coefficients(gram: torch.Tensor, priorities: torch.Tensor,
                                  identifiable: torch.Tensor, headroom: torch.Tensor,
                                  saturation_tolerance: float,
                                  ridge: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an active-ascent direction and project it onto guard half-spaces."""
    active = identifiable & (headroom > saturation_tolerance)
    if not active.any() and identifiable.any():
        identifiable_ids = torch.nonzero(identifiable, as_tuple=False).flatten()
        score = priorities[identifiable_ids] * torch.diag(gram)[identifiable_ids].clamp_min(0).sqrt()
        active[identifiable_ids[torch.argmax(score)]] = True

    coefficients = _simplex_min_norm_coefficients(
        gram=gram, priorities=priorities, active=active, ridge=ridge
    )
    if coefficients.sum().item() <= ridge:
        # No objective has estimable signal.  The resulting advantage remains
        # zero when every component gradient proxy is zero.
        coefficients.fill_(1.0 / max(len(coefficients), 1))

    # Alternating projection onto g_k^T d >= 0.  In coefficient space a
    # projection adds a non-negative multiple of the violated guard gradient.
    gram64 = gram.to(torch.float64)
    for _ in range(4 * max(len(coefficients), 1)):
        margins = gram64 @ coefficients
        violations = torch.nonzero(margins < -ridge, as_tuple=False).flatten()
        if violations.numel() == 0:
            break
        for objective_id in violations.tolist():
            norm_sq = gram64[objective_id, objective_id]
            if norm_sq.item() > ridge:
                coefficients[objective_id] += -margins[objective_id] / norm_sq

    if (gram64 @ coefficients).min().item() < -1e-7:
        nonzero = torch.diag(gram64) > ridge
        coefficients = _simplex_min_norm_coefficients(
            gram=gram64,
            priorities=priorities,
            active=nonzero,
            ridge=ridge,
        )

    coefficient_sum = coefficients.sum()
    if coefficient_sum.item() > ridge:
        coefficients = coefficients / coefficient_sum
    return coefficients.to(gram.dtype), active


def _stable_crossfit_folds(index, fold_count: int) -> tuple[torch.Tensor, list]:
    """Assign complete prompt groups to deterministic, balanced folds."""
    import hashlib

    groups = _group_positions(index)
    if fold_count < 2:
        raise ValueError('SC-PPO cross-fitting requires at least two folds')
    if len(groups) < fold_count:
        raise ValueError('SC-PPO requires at least one prompt group per cross-fit fold')
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(str(item[0]).encode('utf-8')).digest(),
    )
    fold_ids = torch.empty(len(index), dtype=torch.long)
    ordered_positions = []
    for rank, (_, positions) in enumerate(ranked_groups):
        fold_id = rank % fold_count
        fold_ids[torch.as_tensor(positions, dtype=torch.long)] = fold_id
        ordered_positions.append(positions)
    return fold_ids, ordered_positions


def _proxy_gram(component_advantages: torch.Tensor, response_mask: torch.Tensor,
                sample_selector: torch.Tensor) -> torch.Tensor:
    selected_mask = response_mask.to(torch.float64) * sample_selector[:, None].to(torch.float64)
    denominator = selected_mask.sum().clamp_min(1.0)
    values = component_advantages.to(torch.float64) * selected_mask[None, :, :]
    flat_values = values.reshape(values.shape[0], -1)
    return (flat_values @ flat_values.T) / denominator


def _scppo_coefficients_from_statistics(
        gram: torch.Tensor,
        reward_means: torch.Tensor,
        coverage: torch.Tensor,
        targets: Sequence[float],
        priority_floor: float,
        priority_power: float,
        signal_min_group_fraction: float,
        signal_min_relative_norm: float,
        saturation_tolerance: float,
        ridge: float,
        reward_bounds: Sequence[Sequence[float]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norms = torch.diag(gram).clamp_min(0.0).sqrt()
    targets_tensor = torch.as_tensor(targets, dtype=torch.float64, device=gram.device)
    if reward_bounds is not None:
        bounds = torch.as_tensor(reward_bounds, dtype=torch.float64, device=gram.device)
        if bounds.shape != (len(targets), 2) or not torch.isfinite(bounds).all():
            raise ValueError('SC-PPO reward bounds must be a finite [objective, 2] matrix')
        widths = bounds[:, 1] - bounds[:, 0]
        if torch.any(widths <= ridge):
            raise ValueError('SC-PPO reward bounds must have positive width')
        tolerance = max(float(ridge), 1e-6)
        if (torch.any(reward_means < bounds[:, 0] - tolerance) or
                torch.any(reward_means > bounds[:, 1] + tolerance) or
                torch.any(targets_tensor < bounds[:, 0] - tolerance) or
                torch.any(targets_tensor > bounds[:, 1] + tolerance)):
            raise ValueError('SC-PPO reward means and targets must lie inside reward bounds')
        reward_means = ((reward_means - bounds[:, 0]) / widths).clamp(0.0, 1.0)
        targets_tensor = ((targets_tensor - bounds[:, 0]) / widths).clamp(0.0, 1.0)
    headroom = (targets_tensor - reward_means).clamp(0.0, 1.0)
    priorities = headroom.clamp_min(priority_floor).pow(priority_power)
    priorities = priorities / priorities.mean().clamp_min(ridge)
    relative_norm = norms / norms.max().clamp_min(ridge)
    identifiable = (
        (coverage >= signal_min_group_fraction) &
        (relative_norm >= signal_min_relative_norm) &
        (norms > np.sqrt(ridge))
    )
    coefficients, active = _signal_conflict_coefficients(
        gram=gram,
        priorities=priorities,
        identifiable=identifiable,
        headroom=headroom,
        saturation_tolerance=saturation_tolerance,
        ridge=ridge,
    )
    return coefficients, active, priorities


def compute_scppo_calibration_inputs(
        reward_components: Sequence[torch.Tensor],
        eos_mask: torch.Tensor,
        index,
        targets: Sequence[float],
        priority_floor: float,
        priority_power: float,
        signal_min_group_fraction: float,
        signal_min_relative_norm: float,
        saturation_tolerance: float,
        ridge: float,
        objective_centering_means: Sequence[float] = None,
        reward_bounds: Sequence[Sequence[float]] = None) -> tuple[torch.Tensor, dict]:
    """Build per-objective advantages and proxy statistics for gradient audit.

    The returned advantages have shape ``[batch, objective, response]`` so they
    can travel through ``DataProto`` and its data-parallel sharding unchanged.
    Coefficients are diagnostics only; the calibration caller must not use them
    for the shadow update on the same samples.
    """
    if len(reward_components) < 2 or len(targets) != len(reward_components):
        raise ValueError('SC-PPO calibration requires matching multi-reward targets')
    reference = reward_components[0]
    if reference.ndim != 2 or eos_mask.shape != reference.shape:
        raise ValueError('Calibration rewards and response mask must share shape')
    if any(component.shape != reference.shape for component in reward_components):
        raise ValueError('Calibration reward components must share shape')
    if len(index) != reference.shape[0]:
        raise ValueError('Calibration group index length must equal batch size')

    with torch.no_grad():
        component_advantages = torch.stack([
            compute_grpo_outcome_advantage(component, eos_mask, index)[0]
            for component in reward_components
        ], dim=0).to(torch.float64)
        # The treatment whitens the weighted sum across the complete accepted
        # batch. Center each objective with the identical token mask so any
        # weighted combination differs from the actor advantage only by one
        # positive scalar; gradient direction and Gram coefficients then match.
        calibration_mask = eos_mask.to(torch.float64)
        if objective_centering_means is None:
            objective_means = (
                component_advantages * calibration_mask[None, :, :]
            ).sum(dim=(1, 2)) / calibration_mask.sum().clamp_min(1.0)
        else:
            objective_means = torch.as_tensor(
                objective_centering_means,
                dtype=torch.float64,
                device=reference.device,
            )
            if (objective_means.shape != (len(reward_components),) or
                    not torch.isfinite(objective_means).all()):
                raise ValueError('Objective centering means must be a finite objective vector')
        component_advantages = (
            component_advantages - objective_means[:, None, None]
        ) * calibration_mask[None, :, :]
        native_scores = torch.stack([
            component.to(torch.float64).sum(dim=-1) for component in reward_components
        ], dim=0)
        groups = list(_group_positions(index).values())
        if len(groups) < 2:
            raise ValueError('Calibration needs at least two complete prompt groups')

        coverage_values = []
        for objective_id in range(len(reward_components)):
            varying = 0
            for positions in groups:
                position_tensor = torch.as_tensor(
                    positions, dtype=torch.long, device=reference.device
                )
                group_scores = native_scores[objective_id].index_select(0, position_tensor)
                varying += int((group_scores.max() - group_scores.min()).item() > ridge)
            coverage_values.append(varying / len(groups))
        coverage = torch.as_tensor(
            coverage_values, dtype=torch.float64, device=reference.device
        )
        selector = torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device)
        proxy_gram = _proxy_gram(component_advantages, eos_mask, selector)
        reward_means = native_scores.mean(dim=1)
        coefficients, active, priorities = _scppo_coefficients_from_statistics(
            gram=proxy_gram,
            reward_means=reward_means,
            coverage=coverage,
            targets=targets,
            priority_floor=priority_floor,
            priority_power=priority_power,
            signal_min_group_fraction=signal_min_group_fraction,
            signal_min_relative_norm=signal_min_relative_norm,
            saturation_tolerance=saturation_tolerance,
            ridge=ridge,
            reward_bounds=reward_bounds,
        )
        statistics = {
            'proxy_gram': proxy_gram.cpu().tolist(),
            'reward_means': reward_means.cpu().tolist(),
            'coverage': coverage.cpu().tolist(),
            'proxy_coefficients': coefficients.to(torch.float64).cpu().tolist(),
            'proxy_active': active.to(torch.int64).cpu().tolist(),
            'priorities': priorities.cpu().tolist(),
            'objective_centering_means': objective_means.cpu().tolist(),
            'group_count': len(groups),
        }
        if reward_bounds is not None:
            bounds = torch.as_tensor(reward_bounds, dtype=torch.float64, device=reference.device)
            statistics['normalized_reward_means'] = (
                (reward_means - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            ).clamp(0.0, 1.0).cpu().tolist()
        objective_advantages = component_advantages.permute(1, 0, 2).to(torch.float32)
    return objective_advantages, statistics


def scppo_coefficients_from_gram(
        gram,
        reward_means,
        coverage,
        targets: Sequence[float],
        priority_floor: float,
        priority_power: float,
        signal_min_group_fraction: float,
        signal_min_relative_norm: float,
        saturation_tolerance: float,
        ridge: float,
        reward_bounds: Sequence[Sequence[float]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public, CPU-safe coefficient solver used by exact-gradient audits."""
    gram_tensor = torch.as_tensor(gram, dtype=torch.float64)
    gram_tensor = (gram_tensor + gram_tensor.T) / 2.0
    if (gram_tensor.ndim != 2 or gram_tensor.shape[0] != gram_tensor.shape[1] or
            not torch.isfinite(gram_tensor).all()):
        raise ValueError('Gradient Gram must be a finite square matrix')
    diagonal = torch.diag(gram_tensor)
    if diagonal.min().item() < -ridge:
        raise ValueError('Gradient Gram has a materially negative diagonal')
    gram_tensor.diagonal().clamp_min_(0.0)
    return _scppo_coefficients_from_statistics(
        gram=gram_tensor,
        reward_means=torch.as_tensor(reward_means, dtype=torch.float64),
        coverage=torch.as_tensor(coverage, dtype=torch.float64),
        targets=targets,
        priority_floor=priority_floor,
        priority_power=priority_power,
        signal_min_group_fraction=signal_min_group_fraction,
        signal_min_relative_norm=signal_min_relative_norm,
        saturation_tolerance=saturation_tolerance,
        ridge=ridge,
        reward_bounds=reward_bounds,
    )


def coefficient_direction_cosine(left, right, metric_gram, ridge: float = 1e-12) -> float:
    """Cosine between two coefficient directions in a supplied gradient metric."""
    left_tensor = torch.as_tensor(left, dtype=torch.float64)
    right_tensor = torch.as_tensor(right, dtype=torch.float64)
    gram_tensor = torch.as_tensor(metric_gram, dtype=torch.float64)
    numerator = left_tensor @ gram_tensor @ right_tensor
    left_norm = (left_tensor @ gram_tensor @ left_tensor).clamp_min(0.0).sqrt()
    right_norm = (right_tensor @ gram_tensor @ right_tensor).clamp_min(0.0).sqrt()
    denominator = left_norm * right_norm
    if denominator.item() <= ridge:
        return 1.0 if (left_tensor - right_tensor).abs().max().item() <= ridge else 0.0
    return (numerator / denominator).clamp(-1.0, 1.0).item()


def compute_lagged_scppo_outcome_advantage(
        reward_components: Sequence[torch.Tensor],
        eos_mask: torch.Tensor,
        index,
        coefficients: Sequence[float],
        component_names: Sequence[str],
        epsilon: float = 1e-6):
    """Combine independently normalized objectives with detached lagged weights."""
    objective_count = len(reward_components)
    if objective_count < 2 or len(component_names) != objective_count:
        raise ValueError('Lagged SC-PPO requires at least two named objectives')
    reference = reward_components[0]
    if reference.ndim != 2 or eos_mask.shape != reference.shape:
        raise ValueError('Lagged SC-PPO rewards and response mask must share shape')
    if any(component.shape != reference.shape for component in reward_components):
        raise ValueError('Lagged SC-PPO reward components must share shape')
    if len(index) != reference.shape[0]:
        raise ValueError('Lagged SC-PPO group index length must equal batch size')

    weights = torch.as_tensor(coefficients, dtype=torch.float64, device=reference.device)
    if weights.shape != (objective_count,) or not torch.isfinite(weights).all():
        raise ValueError('Lagged SC-PPO coefficients must be a finite objective vector')
    if (weights < 0).any() or not torch.isclose(
            weights.sum(), weights.new_tensor(1.0), atol=1e-10, rtol=0.0):
        raise ValueError('Lagged SC-PPO coefficients must be non-negative and sum to one')

    with torch.no_grad():
        component_advantages = torch.stack([
            compute_grpo_outcome_advantage(component, eos_mask, index, epsilon=epsilon)[0]
            for component in reward_components
        ], dim=0)
        # Multiplication by K is immaterial after whitening and makes the fixed
        # uniform prior exactly update-equivalent to GDPO's unit weights.
        effective_weights = weights.to(component_advantages.dtype) * objective_count
        raw_direction = (
            component_advantages * effective_weights[:, None, None]
        ).sum(dim=0)
        advantages = verl_F.masked_whiten(raw_direction, eos_mask) * eos_mask
        diagnostics = {
            'scppo_treatment/group_count': float(len(_group_positions(index))),
            'scppo_treatment/coefficient_sum': weights.sum().item(),
        }
        for objective_id, component_name in enumerate(component_names):
            diagnostics[f'scppo_treatment/coefficient/{component_name}'] = weights[objective_id].item()
            diagnostics[f'scppo_treatment/proxy_margin/{component_name}'] = _masked_inner_product(
                component_advantages[objective_id], raw_direction, eos_mask
            ).item()
    return advantages, advantages, diagnostics


def prompt_bootstrap_directional_lcbs(
        objective_advantages: torch.Tensor,
        response_mask: torch.Tensor,
        index,
        coefficients: Sequence[float],
        replicates: int,
        seed: int,
        quantile: float = 0.05) -> tuple[list[float], list[float]]:
    """Return prompt-cluster directional margins and one-sided bootstrap LCBs."""
    if objective_advantages.ndim != 3:
        raise ValueError('Objective advantages must have shape [batch, objective, response]')
    batch_size, objective_count, response_length = objective_advantages.shape
    if response_mask.shape != (batch_size, response_length) or len(index) != batch_size:
        raise ValueError('Bootstrap objective advantages, mask, and groups are misaligned')
    weights = torch.as_tensor(coefficients, dtype=torch.float64)
    if weights.shape != (objective_count,) or (weights < 0).any() or not torch.isclose(
            weights.sum(), weights.new_tensor(1.0), atol=1e-10, rtol=0.0):
        raise ValueError('Bootstrap coefficients must be non-negative and sum to one')
    if replicates < 1 or not 0 < quantile < 0.5:
        raise ValueError('Bootstrap replicates and lower-tail quantile are invalid')

    values = objective_advantages.detach().to(device='cpu', dtype=torch.float64)
    mask = response_mask.detach().to(device='cpu', dtype=torch.float64)
    direction = (values * weights[None, :, None]).sum(dim=1)
    group_margins = []
    for positions in _group_positions(index).values():
        positions_tensor = torch.as_tensor(positions, dtype=torch.long)
        group_mask = mask.index_select(0, positions_tensor)
        denominator = group_mask.sum().clamp_min(1.0)
        objective_values = values.index_select(0, positions_tensor)
        direction_values = direction.index_select(0, positions_tensor)
        group_margins.append(
            (objective_values * direction_values[:, None, :] * group_mask[:, None, :])
            .sum(dim=(0, 2)) / denominator
        )
    if len(group_margins) < 2:
        raise ValueError('Prompt bootstrap requires at least two groups')
    margin_matrix = torch.stack(group_margins).numpy()
    generator = np.random.default_rng(int(seed))
    sampled_groups = generator.integers(
        0, margin_matrix.shape[0],
        size=(int(replicates), margin_matrix.shape[0]),
        dtype=np.int32,
    )
    bootstrap_means = margin_matrix[sampled_groups].mean(axis=1)
    means = margin_matrix.mean(axis=0)
    lower_bounds = np.quantile(bootstrap_means, quantile, axis=0, method='linear')
    return means.astype(np.float64).tolist(), lower_bounds.astype(np.float64).tolist()


def _leave_one_group_out_directions(
        component_advantages: torch.Tensor,
        native_scores: torch.Tensor,
        response_mask: torch.Tensor,
        groups: Sequence[Sequence[int]],
        targets: Sequence[float],
        priority_floor: float,
        priority_power: float,
        signal_min_group_fraction: float,
        signal_min_relative_norm: float,
        saturation_tolerance: float,
        ridge: float,
        reward_bounds: Sequence[Sequence[float]] = None):
    """Cross-fit each prompt group against sufficient statistics of all others."""
    objective_count, batch_size, _ = component_advantages.shape
    if len(groups) < 2:
        raise ValueError('SC-PPO leave-one-group-out requires at least two prompt groups')
    gram_numerators = []
    token_counts = []
    score_sums = []
    response_counts = []
    varying = []
    position_tensors = []
    for positions in groups:
        position_tensor = torch.as_tensor(
            positions, dtype=torch.long, device=response_mask.device
        )
        position_tensors.append(position_tensor)
        group_mask = response_mask.index_select(0, position_tensor).to(torch.float64)
        group_values = (
            component_advantages.index_select(1, position_tensor).to(torch.float64) *
            group_mask[None, :, :]
        ).reshape(objective_count, -1)
        gram_numerators.append(group_values @ group_values.T)
        token_counts.append(group_mask.sum())
        group_scores = native_scores.index_select(1, position_tensor).to(torch.float64)
        score_sums.append(group_scores.sum(dim=1))
        response_counts.append(len(positions))
        varying.append((group_scores.max(dim=1).values - group_scores.min(dim=1).values) > ridge)

    stacked_grams = torch.stack(gram_numerators)
    stacked_tokens = torch.stack(token_counts)
    stacked_scores = torch.stack(score_sums)
    stacked_varying = torch.stack(varying).to(torch.float64)
    total_gram = stacked_grams.sum(dim=0)
    total_tokens = stacked_tokens.sum()
    total_scores = stacked_scores.sum(dim=0)
    total_varying = stacked_varying.sum(dim=0)
    raw_direction = torch.zeros(
        (batch_size, response_mask.shape[1]), dtype=torch.float64, device=response_mask.device
    )
    coefficients_per_group = []
    active_per_group = []
    priorities_per_group = []
    coverage_per_group = []

    for group_id, position_tensor in enumerate(position_tensors):
        train_token_count = (total_tokens - stacked_tokens[group_id]).clamp_min(1.0)
        gram = (total_gram - stacked_grams[group_id]) / train_token_count
        train_response_count = batch_size - response_counts[group_id]
        reward_means = (total_scores - stacked_scores[group_id]) / max(train_response_count, 1)
        coverage = (total_varying - stacked_varying[group_id]) / max(len(groups) - 1, 1)
        coefficients, active, priorities = _scppo_coefficients_from_statistics(
            gram=gram,
            reward_means=reward_means,
            coverage=coverage,
            targets=targets,
            priority_floor=priority_floor,
            priority_power=priority_power,
            signal_min_group_fraction=signal_min_group_fraction,
            signal_min_relative_norm=signal_min_relative_norm,
            saturation_tolerance=saturation_tolerance,
            ridge=ridge,
            reward_bounds=reward_bounds,
        )
        group_direction = (
            component_advantages.index_select(1, position_tensor) * coefficients[:, None, None]
        ).sum(dim=0)
        raw_direction.index_copy_(0, position_tensor, group_direction)
        coefficients_per_group.append(coefficients)
        active_per_group.append(active.to(torch.float64))
        priorities_per_group.append(priorities)
        coverage_per_group.append(coverage)

    return (
        raw_direction,
        coefficients_per_group,
        active_per_group,
        priorities_per_group,
        coverage_per_group,
    )


def _group_margin_statistics(component_advantage: torch.Tensor, direction: torch.Tensor,
                             response_mask: torch.Tensor, groups: Sequence[Sequence[int]],
                             confidence_z: float) -> tuple[float, float, float]:
    margins = []
    for positions in groups:
        position_tensor = torch.as_tensor(
            positions, dtype=torch.long, device=response_mask.device
        )
        group_mask = response_mask.index_select(0, position_tensor)
        margins.append(_masked_inner_product(
            component_advantage.index_select(0, position_tensor),
            direction.index_select(0, position_tensor),
            group_mask,
        ))
    stacked = torch.stack(margins).to(torch.float64)
    mean = stacked.mean()
    if len(stacked) > 1:
        standard_error = stacked.std(correction=1) / np.sqrt(len(stacked))
    else:
        standard_error = stacked.new_tensor(float('inf'))
    lower_bound = mean - confidence_z * standard_error
    return mean.item(), standard_error.item(), lower_bound.item()


def compute_scppo_outcome_advantage(
        reward_components: Sequence[torch.Tensor],
        eos_mask: torch.Tensor,
        index,
        component_names: Sequence[str],
        targets: Sequence[float],
        priority_floor: float = 0.05,
        priority_power: float = 0.5,
        signal_min_group_fraction: float = 0.05,
        signal_min_relative_norm: float = 0.05,
        saturation_tolerance: float = 0.05,
        crossfit_folds: int = 2,
        confidence_z: float = 1.6448536269514722,
        ridge: float = 1e-8,
        gdpo_reward_weights: Sequence[float] = None,
        dvao_base_weights: Sequence[float] = None,
        dvao_reward_bounds: Sequence[Sequence[float]] = None,
        dvao_epsilon: float = 1e-6,
        dvao_std_correction: int = 0,
        reward_bounds: Sequence[Sequence[float]] = None,
        shadow: bool = True):
    """Compute SC-PPO or an update-equivalent GDPO shadow diagnostic.

    Geometry is estimated in log-probability/Fisher proxy space.  It is not
    presented as an exact parameter-gradient Gram matrix; the cross-fit folds
    keep each response independent of the coefficients applied to its fold.
    """
    objective_count = len(reward_components)
    if objective_count < 2 or len(component_names) != objective_count:
        raise ValueError('SC-PPO requires at least two named reward components')
    if len(targets) != objective_count or len(set(component_names)) != objective_count:
        raise ValueError('SC-PPO targets and unique names must match reward components')
    if not 0 < priority_floor <= 1 or not 0 < priority_power <= 1:
        raise ValueError('SC-PPO priority_floor and priority_power must lie in (0, 1]')
    if not 0 <= signal_min_group_fraction <= 1 or not 0 <= signal_min_relative_norm <= 1:
        raise ValueError('SC-PPO signal thresholds must lie in [0, 1]')
    if not 0 <= saturation_tolerance <= 1 or confidence_z <= 0 or ridge <= 0:
        raise ValueError('SC-PPO tolerance, confidence_z, and ridge are invalid')
    if int(crossfit_folds) != -1 and int(crossfit_folds) < 2:
        raise ValueError('SC-PPO crossfit_folds must be -1 (leave-one-group-out) or at least 2')

    reference = reward_components[0]
    if reference.ndim != 2 or eos_mask.shape != reference.shape:
        raise ValueError('SC-PPO rewards and response mask must share shape (batch, response)')
    if any(component.shape != reference.shape for component in reward_components):
        raise ValueError('All SC-PPO reward components must have identical shapes')
    if len(index) != reference.shape[0]:
        raise ValueError('SC-PPO group index length must equal batch size')

    with torch.no_grad():
        component_advantages = torch.stack([
            compute_grpo_outcome_advantage(component, eos_mask, index)[0]
            for component in reward_components
        ], dim=0).to(torch.float64)
        native_scores = torch.stack([
            component.to(torch.float64).sum(dim=-1) for component in reward_components
        ], dim=0)
        if int(crossfit_folds) == -1:
            groups = list(_group_positions(index).values())
            (raw_direction, fold_coefficients, fold_active,
             fold_priorities, fold_coverages) = _leave_one_group_out_directions(
                component_advantages=component_advantages,
                native_scores=native_scores,
                response_mask=eos_mask,
                groups=groups,
                targets=targets,
                priority_floor=priority_floor,
                priority_power=priority_power,
                signal_min_group_fraction=signal_min_group_fraction,
                signal_min_relative_norm=signal_min_relative_norm,
                saturation_tolerance=saturation_tolerance,
                ridge=ridge,
                reward_bounds=reward_bounds,
            )
            effective_fold_count = len(groups)
        else:
            fold_ids_cpu, groups = _stable_crossfit_folds(index, int(crossfit_folds))
            fold_ids = fold_ids_cpu.to(reference.device)
            raw_direction = torch.zeros_like(reference, dtype=torch.float64)
            fold_coefficients = []
            fold_active = []
            fold_priorities = []
            fold_coverages = []
            for target_fold in range(int(crossfit_folds)):
                train_selector = fold_ids != target_fold
                target_selector = fold_ids == target_fold
                gram = _proxy_gram(component_advantages, eos_mask, train_selector)
                reward_means = native_scores[:, train_selector].mean(dim=1)
                coverage_values = []
                for objective_id in range(objective_count):
                    varying_groups = 0
                    considered_groups = 0
                    for positions in groups:
                        position_tensor = torch.as_tensor(
                            positions, dtype=torch.long, device=reference.device
                        )
                        if not train_selector[position_tensor[0]]:
                            continue
                        group_scores = native_scores[objective_id].index_select(0, position_tensor)
                        varying_groups += int((group_scores.max() - group_scores.min()).item() > ridge)
                        considered_groups += 1
                    coverage_values.append(varying_groups / max(considered_groups, 1))
                coverage = torch.as_tensor(
                    coverage_values, dtype=torch.float64, device=reference.device
                )
                coefficients, active, priorities = _scppo_coefficients_from_statistics(
                    gram=gram,
                    reward_means=reward_means,
                    coverage=coverage,
                    targets=targets,
                    priority_floor=priority_floor,
                    priority_power=priority_power,
                    signal_min_group_fraction=signal_min_group_fraction,
                    signal_min_relative_norm=signal_min_relative_norm,
                    saturation_tolerance=saturation_tolerance,
                    ridge=ridge,
                    reward_bounds=reward_bounds,
                )
                fold_coefficients.append(coefficients)
                fold_active.append(active.to(torch.float64))
                fold_priorities.append(priorities)
                fold_coverages.append(coverage)
                fold_direction = (component_advantages * coefficients[:, None, None]).sum(dim=0)
                raw_direction[target_selector] = fold_direction[target_selector]
            effective_fold_count = int(crossfit_folds)

        scppo_advantages = verl_F.masked_whiten(raw_direction.to(torch.float32), eos_mask) * eos_mask
        gdpo_advantages, _ = compute_gdpo_outcome_advantage(
            reward_components=reward_components,
            eos_mask=eos_mask,
            index=index,
            reward_weights=gdpo_reward_weights,
        )
        grpo_advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=sum(reward_components),
            eos_mask=eos_mask,
            index=index,
        )
        dvao_advantages, _, _ = compute_dvao_outcome_advantage(
            reward_components=reward_components,
            eos_mask=eos_mask,
            index=index,
            base_weights=dvao_base_weights,
            reward_bounds=dvao_reward_bounds,
            component_names=component_names,
            epsilon=dvao_epsilon,
            std_correction=dvao_std_correction,
        )

        diagnostics = {
            'scppo/shadow/update_is_gdpo': float(bool(shadow)),
            'scppo/group_count': float(len(groups)),
            'scppo/crossfit_folds': float(effective_fold_count),
            'scppo/crossfit_leave_one_group_out': float(int(crossfit_folds) == -1),
        }
        stacked_coefficients = torch.stack(fold_coefficients)
        stacked_active = torch.stack(fold_active)
        stacked_priorities = torch.stack(fold_priorities)
        stacked_coverages = torch.stack(fold_coverages)
        full_reward_means = native_scores.mean(dim=1)
        for objective_id, name in enumerate(component_names):
            diagnostics[f'scppo/reward/{name}_mean'] = full_reward_means[objective_id].item()
            diagnostics[f'scppo/signal/{name}_mixed_group_fraction'] = stacked_coverages[:, objective_id].mean().item()
            diagnostics[f'scppo/priority/{name}_mean'] = stacked_priorities[:, objective_id].mean().item()
            diagnostics[f'scppo/active/{name}_fold_fraction'] = stacked_active[:, objective_id].mean().item()
            diagnostics[f'scppo/coefficient/{name}_mean'] = stacked_coefficients[:, objective_id].mean().item()
        diagnostics['scppo/coefficient/fold_l1_disagreement'] = (
            (stacked_coefficients - stacked_coefficients.mean(dim=0)).abs().sum(dim=1).mean().item()
        )

        methods = {
            'grpo': grpo_advantages,
            'gdpo': gdpo_advantages,
            'dvao': dvao_advantages,
            'scppo': scppo_advantages,
        }
        for method_name, direction in methods.items():
            lower_bounds = []
            for objective_id, component_name in enumerate(component_names):
                mean, standard_error, lower_bound = _group_margin_statistics(
                    component_advantage=component_advantages[objective_id],
                    direction=direction,
                    response_mask=eos_mask,
                    groups=groups,
                    confidence_z=confidence_z,
                )
                diagnostics[f'scppo/proxy/{method_name}/margin/{component_name}'] = mean
                diagnostics[f'scppo/proxy/{method_name}/se/{component_name}'] = standard_error
                diagnostics[f'scppo/proxy/{method_name}/lcb95/{component_name}'] = lower_bound
                lower_bounds.append(lower_bound)
            diagnostics[f'scppo/proxy/{method_name}/min_lcb95'] = min(lower_bounds)
            direction_norm = _masked_inner_product(direction, direction, eos_mask).clamp_min(ridge).sqrt()
            gdpo_norm = _masked_inner_product(gdpo_advantages, gdpo_advantages, eos_mask).clamp_min(ridge).sqrt()
            diagnostics[f'scppo/proxy/{method_name}/cosine_to_gdpo'] = (
                _masked_inner_product(direction, gdpo_advantages, eos_mask) /
                (direction_norm * gdpo_norm)
            ).item()

        chosen_advantages = gdpo_advantages if shadow else scppo_advantages
    return chosen_advantages, chosen_advantages, diagnostics


def compute_dvao_outcome_advantage(
        reward_components: Sequence[torch.Tensor],
        eos_mask: torch.Tensor,
        index,
        base_weights: Sequence[float],
        reward_bounds: Sequence[Sequence[float]],
        component_names: Sequence[str] = None,
        epsilon: float = 1e-6,
        std_correction: int = 0):
    """DVAO Eq. (8): variance-adaptive combination without batch whitening."""
    if len(reward_components) < 2:
        raise ValueError('DVAO requires at least two reward components')
    if len(reward_components) != len(base_weights) or len(reward_components) != len(reward_bounds):
        raise ValueError('DVAO components, weights, and bounds must have equal length')
    if component_names is None:
        component_names = [f'reward_{i}' for i in range(len(reward_components))]
    if len(component_names) != len(reward_components) or len(set(component_names)) != len(component_names):
        raise ValueError('DVAO component names must be unique and match reward components')
    if epsilon <= 0 or std_correction != 0:
        raise ValueError('DVAO requires epsilon > 0 and population std_correction == 0')

    reference = reward_components[0]
    if reference.ndim != 2 or eos_mask.shape != reference.shape:
        raise ValueError('DVAO rewards and response mask must share shape (batch, response)')
    if any(component.shape != reference.shape or component.device != reference.device
           for component in reward_components):
        raise ValueError('All DVAO reward components must share shape and device')
    if len(index) != reference.shape[0]:
        raise ValueError('DVAO group index length must equal batch size')

    weights = torch.as_tensor(base_weights, dtype=torch.float32, device=reference.device)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('DVAO base weights must be finite and non-negative')
    if not torch.isclose(weights.sum(), weights.new_tensor(1.0), atol=1e-6):
        raise ValueError(f'DVAO base weights must sum to 1, got {weights.sum().item()}')
    bounds = torch.as_tensor(reward_bounds, dtype=torch.float32, device=reference.device)
    if bounds.shape != (len(reward_components), 2):
        raise ValueError('DVAO reward bounds must have shape (components, 2)')
    if not torch.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
        raise ValueError('Every DVAO reward bound must satisfy finite lower < upper')

    with torch.no_grad():
        native_scores = torch.stack(
            [component.to(torch.float32).sum(dim=-1) for component in reward_components], dim=0
        )
        tolerance = 1e-5
        invalid = (native_scores < bounds[:, 0:1] - tolerance) | (
            native_scores > bounds[:, 1:2] + tolerance
        )
        if invalid.any():
            component_id, sample_id = torch.nonzero(invalid, as_tuple=False)[0].tolist()
            raise ValueError(
                f'DVAO reward {component_names[component_id]} at sample {sample_id} is outside bounds'
            )
        scores = (native_scores - bounds[:, 0:1]) / (bounds[:, 1:2] - bounds[:, 0:1])

        sequence_advantages = torch.zeros(reference.shape[0], dtype=torch.float32, device=reference.device)
        group_stds = []
        group_dynamic_weights = []
        group_denominators = []
        active_groups = 0
        for positions in _group_positions(index).values():
            position_tensor = torch.as_tensor(positions, dtype=torch.long, device=reference.device)
            group_scores = scores.index_select(1, position_tensor)
            means = group_scores.mean(dim=1, keepdim=True)
            stds = torch.std(group_scores, dim=1, correction=std_correction)
            weighted_stds = weights * stds
            denominator = weighted_stds.sum()
            if denominator.item() > epsilon:
                group_advantages = (weights[:, None] * (group_scores - means)).sum(dim=0) / denominator
                dynamic_weights = weighted_stds / denominator
                active_groups += 1
            else:
                group_advantages = torch.zeros(len(positions), dtype=torch.float32, device=reference.device)
                dynamic_weights = torch.zeros_like(weights)
            sequence_advantages.index_copy_(0, position_tensor, group_advantages)
            group_stds.append(stds)
            group_dynamic_weights.append(dynamic_weights)
            group_denominators.append(denominator)

        advantages = sequence_advantages.unsqueeze(-1).expand_as(eos_mask) * eos_mask
        stacked_stds = torch.stack(group_stds)
        stacked_weights = torch.stack(group_dynamic_weights)
        diagnostics = {
            'dvao/group_count': float(len(group_stds)),
            'dvao/active_group_fraction': active_groups / max(len(group_stds), 1),
            'dvao/denominator_mean': torch.stack(group_denominators).mean().item(),
            'dvao/sequence_advantage_abs_mean': sequence_advantages.abs().mean().item(),
            'dvao/sequence_advantage_abs_max': sequence_advantages.abs().max().item(),
        }
        for component_id, name in enumerate(component_names):
            diagnostics[f'dvao/std/{name}_mean'] = stacked_stds[:, component_id].mean().item()
            diagnostics[f'dvao/weight/{name}_all_groups_mean'] = stacked_weights[:, component_id].mean().item()

    return advantages, advantages, diagnostics


def compute_grpo_no_std_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

# compute_grpo_bn_outcome_advantage
def compute_grpo_bn_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
        scores = verl_F.masked_whiten(scores, eos_mask) * eos_mask

    return scores, scores


def compute_grpo_no_std_bn_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
        scores = verl_F.masked_whiten(scores, eos_mask) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        eos_mask,
        cliprange,
        cliprange_low=None,
        cliprange_high=None,
        loss_agg_mode='seq-mean-token-mean',
        normalization_factor=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1.0 - cliprange_low, 1.0 + cliprange_high
    )

    loss_matrix = torch.max(pg_losses, pg_losses2)
    pg_loss = aggregate_loss(
        loss_matrix=loss_matrix,
        eos_mask=eos_mask,
        loss_agg_mode=loss_agg_mode,
        normalization_factor=normalization_factor,
    )
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def aggregate_loss(loss_matrix, eos_mask, loss_agg_mode, normalization_factor=None):
    """Aggregate a micro-batch loss with DAPO's global token normalization."""
    if loss_matrix.shape != eos_mask.shape:
        raise ValueError('loss matrix and response mask must have identical shapes')
    if loss_agg_mode == 'token-mean':
        if normalization_factor is None:
            normalization_factor = 1.0 / eos_mask.sum().clamp_min(1)
        return verl_F.masked_sum(loss_matrix, eos_mask) * normalization_factor
    if loss_agg_mode == 'seq-mean-token-mean':
        token_counts = eos_mask.sum(dim=-1)
        valid_sequences = (token_counts > 0).to(eos_mask.dtype)
        sequence_losses = (loss_matrix * eos_mask).sum(dim=-1) / token_counts.clamp_min(1)
        return (sequence_losses * valid_sequences).sum() / valid_sequences.sum().clamp_min(1)
    raise ValueError(f'Unsupported policy loss aggregation mode: {loss_agg_mode}')


def combine_policy_losses(pg_loss, entropy_loss, entropy_coeff, kl_loss=None, kl_loss_coef=0.0):
    """Compose the minimized actor objective; KL is a positive regularizer."""
    policy_loss = pg_loss - entropy_loss * entropy_coeff
    if kl_loss is not None:
        policy_loss = policy_loss + kl_loss * kl_loss_coef
    return policy_loss


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
