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
Single Process Actor
"""

import itertools
import json
import os
import time
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


def _response_logit_indices(unpad_indices: torch.Tensor, seqlen: int, response_length: int) -> torch.Tensor:
    """Select unpadded hidden states whose next-token logits can enter the PPO loss."""
    columns = torch.remainder(unpad_indices, seqlen)
    first_logit_column = seqlen - response_length - 1
    last_logit_column = seqlen - 1
    return torch.nonzero(
        (columns >= first_logit_column) & (columns < last_logit_column),
        as_tuple=False,
    ).flatten()


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        actor_lr_scheduler=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            verl_F.entropy_from_logits
            if os.getenv('VERL_DISABLE_TORCH_COMPILE', '0') == '1'
            else torch.compile(verl_F.entropy_from_logits, dynamic=True)
        )
        self._lm_head_indices = None
        unwrapped_actor = actor_module.module if isinstance(actor_module, FSDP) else actor_module
        lm_head = unwrapped_actor.get_output_embeddings()
        if lm_head is None:
            raise ValueError('The actor model must expose output embeddings for selective logits')

        def select_lm_head_inputs(_module, inputs):
            if self._lm_head_indices is None:
                return None
            return (inputs[0].index_select(1, self._lm_head_indices), *inputs[1:])

        self._lm_head_hook = lm_head.register_forward_pre_hook(select_lm_head_inputs)

        moco_config = self.config.get('moco_cagrad_shadow', {})
        self.moco_cagrad_shadow_enabled = bool(moco_config.get('enabled', False))
        self.moco_cagrad_apply_update = bool(moco_config.get('apply_update', False))
        self.moco_cagrad_tracker = None
        self.moco_cagrad_execute = None
        self.moco_cagrad_host_preflight = None
        self.moco_cagrad_host_preflight_fn = None
        self.moco_cagrad_minimum_host_headroom_gib = None
        self.moco_cagrad_allocation_overhead_factor = None
        if self.moco_cagrad_shadow_enabled:
            if self.actor_optimizer is None or self.actor_lr_scheduler is None:
                raise RuntimeError('MoCo--CAGrad shadow requires actor optimizer and scheduler')
            from moco_cagrad_shadow import (
                FullVectorTracker,
                execute_readonly_shadow_step,
                prepare_active_update,
                tracker_host_memory_preflight,
            )
            self.moco_cagrad_host_preflight_fn = tracker_host_memory_preflight
            self.moco_cagrad_minimum_host_headroom_gib = float(
                moco_config.get('minimum_host_headroom_gib', 8.0)
            )
            self.moco_cagrad_allocation_overhead_factor = float(
                moco_config.get('allocation_overhead_factor', 1.10)
            )
            self.moco_cagrad_tracker = FullVectorTracker(
                objective_count=int(moco_config.objective_count),
                horizon_actor_updates=int(moco_config.horizon_actor_updates),
                maximum_norm=float(moco_config.maximum_norm),
                radius=float(moco_config.radius),
                resolution=int(moco_config.resolution),
                gather_reference_steps=tuple(
                    int(step) for step in moco_config.gather_reference_steps
                ),
                diagnostic_steps=(
                    None
                    if moco_config.get('diagnostic_steps') is None
                    else tuple(int(step) for step in moco_config.diagnostic_steps)
                ),
                gather_chunk_elements=int(moco_config.gather_chunk_elements),
            )
            self.moco_cagrad_execute = (
                prepare_active_update
                if self.moco_cagrad_apply_update
                else execute_readonly_shadow_step
            )

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=True,
        entropy_requires_grad=True,
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                selected_indices = None
                selected_pad_indices = indices
                if not self.use_ulysses_sp:
                    selected_indices = _response_logit_indices(indices, seqlen, response_length)
                    selected_pad_indices = indices.index_select(0, selected_indices)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.index_select(0, selected_indices)
                    self._lm_head_indices = selected_indices

                # only pass input_ids and position_ids to enable flash_attn_varlen
                try:
                    output = self.actor_module(input_ids=input_ids_rmpad,
                                               attention_mask=None,
                                               position_ids=position_ids_rmpad,
                                               use_cache=False)  # prevent model thinks we are generating
                finally:
                    self._lm_head_indices = None
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                entropy_rmpad = None
                if calculate_entropy:
                    # Entropy is still reported when its coefficient is zero,
                    # but its softmax graph must not occupy backward memory.
                    entropy_logits = logits_rmpad if entropy_requires_grad else logits_rmpad.detach()
                    entropy_rmpad = self.compute_entropy_from_logits(entropy_logits)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if entropy_rmpad is not None:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=selected_pad_indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = None
                if entropy_rmpad is not None:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=selected_pad_indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False,
                                           num_logits_to_keep=response_length + 1)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = None
                if calculate_entropy:
                    entropy_logits = logits if entropy_requires_grad else logits.detach()
                    entropy = verl_F.entropy_from_logits(entropy_logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            # Keep the logical response batch on CPU and stage only the
            # token-balanced micro-batch needed by this forward pass.
            micro_batch = micro_batch.cuda()
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=False,
                )
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(
                get_reverse_idx(indices), dtype=torch.long, device=log_probs.device
            )
            log_probs = log_probs[revert_indices]

        return log_probs

    def compute_objective_gradient_gram(self, data: DataProto):
        """Measure sharded per-objective policy-gradient geometry without an update.

        ``exact`` retains every FSDP gradient shard and also reports the fixed
        last-block anchor sub-Gram. ``anchor`` temporarily freezes all but the
        registered final transformer blocks, avoiding backward through earlier
        layers. Optimizer and scheduler state are never changed.
        """
        self.actor_module.train()
        if self.actor_optimizer is None:
            raise RuntimeError('Gradient calibration requires an actor optimizer')
        if self.config.get('loss_agg_mode', 'seq-mean-token-mean') != 'token-mean':
            raise ValueError('Exact calibration currently requires token-mean PPO loss')

        gradient_mode = str(data.meta_info.get('gradient_mode', 'anchor'))
        if gradient_mode not in ('anchor', 'exact'):
            raise ValueError(f'Unknown gradient calibration mode: {gradient_mode}')
        anchor_last_n_layers = int(data.meta_info.get('anchor_last_n_layers', 2))
        if anchor_last_n_layers < 1:
            raise ValueError('anchor_last_n_layers must be positive')
        temperature = data.meta_info['temperature']
        select_keys = [
            'responses', 'input_ids', 'attention_mask', 'position_ids',
            'old_log_probs', 'objective_advantages',
        ]
        batch = data.select(batch_keys=select_keys).batch
        objective_count = int(batch['objective_advantages'].shape[1])
        if objective_count < 2 or objective_count > 10:
            raise ValueError('Gradient calibration supports 2--10 objectives')

        model_config = getattr(self.actor_module, 'module', self.actor_module).config
        layer_count = int(model_config.num_hidden_layers)
        anchor_layer_ids = tuple(range(max(0, layer_count - anchor_last_n_layers), layer_count))
        named_parameters = list(self.actor_module.named_parameters())
        cpu_rng_state = torch.get_rng_state().clone()
        cuda_rng_state = torch.cuda.get_rng_state().clone()
        parameter_versions = {name: parameter._version for name, parameter in named_parameters}

        def optimizer_state_signature():
            signature = []
            for parameter, state in self.actor_optimizer.state.items():
                state_items = []
                for key, value in sorted(state.items()):
                    if torch.is_tensor(value):
                        state_items.append((
                            key, value.data_ptr(), tuple(value.shape), str(value.dtype), value._version
                        ))
                    else:
                        state_items.append((key, repr(value)))
                signature.append((id(parameter), tuple(state_items)))
            return tuple(sorted(signature, key=lambda item: item[0]))

        optimizer_signature = optimizer_state_signature()
        anchor_names = {
            name for name, _ in named_parameters
            if any(f'.layers.{layer_id}.' in f'.{name}.' for layer_id in anchor_layer_ids)
        }
        if not anchor_names:
            visible = ', '.join(name for name, _ in named_parameters[:8])
            raise RuntimeError(
                f'No FSDP parameters matched anchor layers {anchor_layer_ids}; first names: {visible}'
            )

        original_requires_grad = {name: parameter.requires_grad for name, parameter in named_parameters}
        if gradient_mode == 'anchor':
            for name, parameter in named_parameters:
                parameter.requires_grad_(name in anchor_names)

        response_length = batch['responses'].size(1)
        response_mask = batch['attention_mask'][:, -response_length:]
        global_response_tokens = response_mask.sum().to(
            device=torch.cuda.current_device(), dtype=torch.float32
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_response_tokens, op=torch.distributed.ReduceOp.SUM)
            data_parallel_size = torch.distributed.get_world_size()
        else:
            data_parallel_size = 1
        if global_response_tokens.item() <= 0:
            raise ValueError('Calibration batch contains no valid response tokens')
        token_normalization = data_parallel_size / global_response_tokens

        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(self.config.ppo_micro_batch_size)

        torch.cuda.reset_peak_memory_stats()
        gradient_snapshots = []
        try:
            for objective_id in range(objective_count):
                self.actor_optimizer.zero_grad(set_to_none=True)
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.cuda()
                    responses = micro_batch['responses']
                    local_response_length = responses.size(1)
                    local_response_mask = micro_batch['attention_mask'][:, -local_response_length:]
                    _, log_prob = self._forward_micro_batch(
                        micro_batch=micro_batch,
                        temperature=temperature,
                        calculate_entropy=False,
                    )
                    pg_loss, _, _ = core_algos.compute_policy_loss(
                        old_log_prob=micro_batch['old_log_probs'],
                        log_prob=log_prob,
                        advantages=micro_batch['objective_advantages'][:, objective_id, :],
                        eos_mask=local_response_mask,
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.get('clip_ratio_low', self.config.clip_ratio),
                        cliprange_high=self.config.get('clip_ratio_high', self.config.clip_ratio),
                        loss_agg_mode='token-mean',
                        normalization_factor=token_normalization,
                    )
                    pg_loss.backward()

                snapshot = {}
                for name, parameter in named_parameters:
                    if parameter.grad is None:
                        continue
                    if gradient_mode == 'anchor' and name not in anchor_names:
                        raise RuntimeError(f'Frozen non-anchor parameter received a gradient: {name}')
                    snapshot[name] = parameter.grad.detach().to(
                        device='cpu', dtype=torch.float32
                    ).reshape(-1).clone()
                if not snapshot:
                    raise RuntimeError(f'Objective {objective_id} produced no gradient shards')
                gradient_snapshots.append(snapshot)

            def local_gram(allowed_names=None):
                gram = torch.zeros((objective_count, objective_count), dtype=torch.float64)
                for left_id in range(objective_count):
                    for right_id in range(left_id, objective_count):
                        dot_product = 0.0
                        common_names = gradient_snapshots[left_id].keys() & gradient_snapshots[right_id].keys()
                        if allowed_names is not None:
                            common_names = common_names & allowed_names
                        for name in common_names:
                            dot_product += torch.dot(
                                gradient_snapshots[left_id][name],
                                gradient_snapshots[right_id][name],
                            ).item()
                        gram[left_id, right_id] = dot_product
                        gram[right_id, left_id] = dot_product
                return gram

            anchor_gram = local_gram(anchor_names)
            exact_gram = local_gram() if gradient_mode == 'exact' else None

            def globalize(local_matrix):
                matrix = local_matrix.to(device=torch.cuda.current_device())
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(matrix, op=torch.distributed.ReduceOp.SUM)
                return matrix.cpu()

            anchor_gram = globalize(anchor_gram)
            if exact_gram is not None:
                exact_gram = globalize(exact_gram)

            anchor_parameter_count = sum(
                gradient_snapshots[0][name].numel()
                for name in gradient_snapshots[0].keys() & anchor_names
            )
            measured_parameter_count = sum(tensor.numel() for tensor in gradient_snapshots[0].values())
            counts = torch.tensor(
                [anchor_parameter_count, measured_parameter_count],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            peak = torch.tensor(
                [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)

            metrics = {
                'scppo_calibration/mode_exact': float(gradient_mode == 'exact'),
                'scppo_calibration/objective_count': float(objective_count),
                'scppo_calibration/anchor_parameter_count': counts[0].item(),
                'scppo_calibration/measured_parameter_count': counts[1].item(),
                'scppo_calibration/peak_allocated_gib': peak[0].item() / 2**30,
                'scppo_calibration/peak_reserved_gib': peak[1].item() / 2**30,
                'scppo_calibration/reserved_headroom_gib': (
                    torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory - peak[1].item()
                ) / 2**30,
            }
            for left_id in range(objective_count):
                for right_id in range(objective_count):
                    metrics[f'scppo_calibration/anchor_gram/{left_id}/{right_id}'] = (
                        anchor_gram[left_id, right_id].item()
                    )
                    if exact_gram is not None:
                        metrics[f'scppo_calibration/exact_gram/{left_id}/{right_id}'] = (
                            exact_gram[left_id, right_id].item()
                        )
            return metrics
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)
            if gradient_mode == 'anchor':
                for name, parameter in named_parameters:
                    parameter.requires_grad_(original_requires_grad[name])
            changed_parameters = [
                name for name, parameter in named_parameters
                if parameter._version != parameter_versions[name]
            ]
            if changed_parameters:
                raise RuntimeError(
                    'Read-only calibration modified actor parameters: '
                    + ', '.join(changed_parameters[:8])
                )
            if optimizer_state_signature() != optimizer_signature:
                raise RuntimeError('Read-only calibration modified optimizer state')
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
            gradient_snapshots.clear()

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        if self.moco_cagrad_shadow_enabled and self.moco_cagrad_host_preflight is None:
            self.moco_cagrad_host_preflight = self.moco_cagrad_host_preflight_fn(
                self.actor_module,
                self.moco_cagrad_tracker.objective_count,
                optimizer=self.actor_optimizer,
                optimizer_offload=bool(
                    self.config.fsdp_config.get('optimizer_offload', False)
                ),
                minimum_headroom_gib=self.moco_cagrad_minimum_host_headroom_gib,
                allocation_overhead_factor=self.moco_cagrad_allocation_overhead_factor,
            )
            print(
                'moco_cagrad_host_memory_preflight=' + json.dumps(
                    self.moco_cagrad_host_preflight, sort_keys=True, allow_nan=False
                ),
                flush=True,
            )

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        moco_outer_step = data.meta_info.get('moco_outer_step')
        moco_uid_digest = data.meta_info.get('moco_uid_digest')

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.moco_cagrad_shadow_enabled:
            select_keys.append('moco_objective_advantages')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        expected_actor_steps = int(
            self.config.get('moco_cagrad_shadow', {}).get('actor_steps_per_outer', 0)
        )
        if self.moco_cagrad_shadow_enabled and len(dataloader) != expected_actor_steps:
            raise RuntimeError(
                f'MoCo--CAGrad expected {expected_actor_steps} actor steps per outer batch; '
                f'observed {len(dataloader)}'
            )
        initial_tracker_step = (
            self.moco_cagrad_tracker.step if self.moco_cagrad_shadow_enabled else 0
        )
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            mini_response_length = mini_batch['responses'].size(1)
            mini_response_mask = mini_batch['attention_mask'][:, -mini_response_length:]
            # The logical mini-batch may be CPU-resident. NCCL reductions
            # require the normalization scalar to be on the local GPU.
            global_response_tokens = mini_response_mask.sum().to(
                device=torch.cuda.current_device(), dtype=torch.float32
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(global_response_tokens, op=torch.distributed.ReduceOp.SUM)
                data_parallel_size = torch.distributed.get_world_size()
            else:
                data_parallel_size = 1
            if global_response_tokens.item() <= 0:
                raise ValueError('PPO mini-batch contains no valid response tokens')
            token_normalization = data_parallel_size / global_response_tokens

            if os.getenv('RLLA_MEMORY_DEBUG', '0') == '1':
                token_sums = [int(item['attention_mask'].sum().item()) for item in micro_batches]
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f'actor micro-batches rank={rank} minibatch={batch_idx} '
                    f'tokens={token_sums} allocated_gib={torch.cuda.memory_allocated() / 2**30:.3f} '
                    f'reserved_gib={torch.cuda.memory_reserved() / 2**30:.3f}',
                    flush=True,
                )

            if self.moco_cagrad_shadow_enabled:
                if self.config.get('loss_agg_mode', 'seq-mean-token-mean') != 'token-mean':
                    raise ValueError('MoCo--CAGrad Stage A requires token-mean PPO loss')
                objective_count = self.moco_cagrad_tracker.objective_count
                if mini_batch['moco_objective_advantages'].shape != (
                        len(mini_batch), objective_count, mini_response_length):
                    raise ValueError('MoCo--CAGrad objective advantages have an invalid shape')

                def backward_moco_objective(objective_id):
                    for shadow_batch in micro_batches:
                        shadow_batch = shadow_batch.cuda()
                        local_response_length = shadow_batch['responses'].size(1)
                        local_response_mask = shadow_batch[
                            'attention_mask'
                        ][:, -local_response_length:]
                        _, shadow_log_prob = self._forward_micro_batch(
                            micro_batch=shadow_batch,
                            temperature=temperature,
                            calculate_entropy=False,
                        )
                        objective_loss, _, _ = core_algos.compute_policy_loss(
                            old_log_prob=shadow_batch['old_log_probs'],
                            log_prob=shadow_log_prob,
                            advantages=shadow_batch[
                                'moco_objective_advantages'
                            ][:, objective_id, :],
                            eos_mask=local_response_mask,
                            cliprange=self.config.clip_ratio,
                            cliprange_low=self.config.get(
                                'clip_ratio_low', self.config.clip_ratio
                            ),
                            cliprange_high=self.config.get(
                                'clip_ratio_high', self.config.clip_ratio
                            ),
                            loss_agg_mode='token-mean',
                            normalization_factor=token_normalization,
                        )
                        objective_loss.backward()

                entropy_coeff = float(self.config.entropy_coeff)
                kl_loss_coeff = (
                    float(self.config.kl_loss_coef) if self.config.use_kl_loss else 0.0
                )

                def backward_moco_shared():
                    for shadow_batch in micro_batches:
                        shadow_batch = shadow_batch.cuda()
                        local_response_length = shadow_batch['responses'].size(1)
                        local_response_mask = shadow_batch[
                            'attention_mask'
                        ][:, -local_response_length:]
                        shadow_entropy, shadow_log_prob = self._forward_micro_batch(
                            micro_batch=shadow_batch,
                            temperature=temperature,
                            calculate_entropy=entropy_coeff != 0.0,
                            entropy_requires_grad=entropy_coeff != 0.0,
                        )
                        shared_loss = shadow_log_prob.new_zeros(())
                        if entropy_coeff != 0.0:
                            entropy_loss = core_algos.aggregate_loss(
                                loss_matrix=shadow_entropy,
                                eos_mask=local_response_mask,
                                loss_agg_mode='token-mean',
                                normalization_factor=token_normalization,
                            )
                            shared_loss = shared_loss - entropy_coeff * entropy_loss
                        if kl_loss_coeff != 0.0:
                            kld = core_algos.kl_penalty(
                                logprob=shadow_log_prob,
                                ref_logprob=shadow_batch['ref_log_prob'],
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = core_algos.aggregate_loss(
                                loss_matrix=kld,
                                eos_mask=local_response_mask,
                                loss_agg_mode='token-mean',
                                normalization_factor=token_normalization,
                            )
                            shared_loss = shared_loss + kl_loss_coeff * kl_loss
                        shared_loss.backward()

                def reduce_sum_cpu(value):
                    if not torch.distributed.is_initialized():
                        return value
                    reduced = value.to(device=torch.cuda.current_device())
                    torch.distributed.all_reduce(
                        reduced, op=torch.distributed.ReduceOp.SUM
                    )
                    return reduced.cpu()

                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized() else 0
                )
                before_allocated = torch.cuda.memory_allocated()
                before_reserved = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                host_bytes_before = self.moco_cagrad_tracker.host_transfer_bytes
                moco_started = time.perf_counter()
                execute_kwargs = dict(
                    module=self.actor_module,
                    optimizer=self.actor_optimizer,
                    scheduler=self.actor_lr_scheduler,
                    tracker=self.moco_cagrad_tracker,
                    backward_objective=backward_moco_objective,
                    backward_shared=(
                        backward_moco_shared
                        if entropy_coeff != 0.0 or kl_loss_coeff != 0.0 else None
                    ),
                    gradient_clip=float(self.config.grad_clip),
                    reduce_sum=reduce_sum_cpu,
                )
                if self.moco_cagrad_apply_update:
                    execute_kwargs['preview_adamw'] = bool(
                        self.config.get('moco_cagrad_shadow', {}).get(
                            'preview_adamw', False
                        )
                    )
                moco_record, _ = self.moco_cagrad_execute(**execute_kwargs)
                active_grad_norm = None
                if self.moco_cagrad_apply_update:
                    try:
                        active_grad_norm = self._optimizer_step()
                    except Exception:
                        self.moco_cagrad_tracker.poisoned = True
                        raise
                    active_grad_norm_value = float(active_grad_norm.detach().item())
                    moco_record.update({
                        'optimizer_step_applied': True,
                        'grad_norm': active_grad_norm_value,
                    })
                torch.cuda.synchronize()
                total_memory = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                peak_reserved = torch.cuda.max_memory_reserved()
                moco_record.update({
                    'rank': rank,
                    'outer_step': int(moco_outer_step),
                    'inner_step': batch_idx + 1,
                    'actor_steps_per_outer': expected_actor_steps,
                    'uid_digest': str(moco_uid_digest),
                    'wall_seconds': time.perf_counter() - moco_started,
                    'memory_allocated_before_gib': before_allocated / 2**30,
                    'memory_reserved_before_gib': before_reserved / 2**30,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'peak_reserved_gib': peak_reserved / 2**30,
                    'reserved_headroom_gib': (total_memory - peak_reserved) / 2**30,
                    'host_transfer_bytes_step': (
                        self.moco_cagrad_tracker.host_transfer_bytes - host_bytes_before
                    ),
                    'host_memory_preflight': self.moco_cagrad_host_preflight,
                })
                metric_prefix = (
                    'moco_active' if self.moco_cagrad_apply_update else 'moco_shadow'
                )
                marker = (
                    'moco_cagrad_active_rank_json='
                    if self.moco_cagrad_apply_update
                    else 'moco_cagrad_shadow_rank_json='
                )
                print(marker + json.dumps(
                    moco_record, sort_keys=True, allow_nan=False
                ), flush=True)
                append_to_dict(metrics, {
                    f'{metric_prefix}/actor_step': float(moco_record['tracker_step']),
                    f'{metric_prefix}/wall_seconds': moco_record['wall_seconds'],
                    f'{metric_prefix}/reserved_headroom_gib': moco_record[
                        'reserved_headroom_gib'
                    ],
                    f'{metric_prefix}/tracker_direction_norm': moco_record[
                        'tracker_direction_norm'
                    ],
                })
                if moco_record['diagnostics_audited']:
                    append_to_dict(metrics, {
                        f'{metric_prefix}/direction_reference_relative_error': moco_record[
                            'direct_reference_relative_error'
                        ],
                        f'{metric_prefix}/bf16_gram_relative_error': moco_record[
                            'bf16_gram_relative_error'
                        ],
                        f'{metric_prefix}/bf16_direction_cosine': moco_record[
                            'bf16_direction_cosine'
                        ],
                        f'{metric_prefix}/bf16_dual_l1_error': moco_record[
                            'bf16_dual_l1_error'
                        ],
                    })
                if self.moco_cagrad_apply_update:
                    append_to_dict(metrics, {
                        'actor/grad_norm': active_grad_norm_value,
                        'moco_active/optimizer_step_applied': 1.0,
                    })
                    continue

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = self.config.get('clip_ratio_low', clip_ratio)
                clip_ratio_high = self.config.get('clip_ratio_high', clip_ratio)
                loss_agg_mode = self.config.get('loss_agg_mode', 'seq-mean-token-mean')
                entropy_coeff = self.config.entropy_coeff
                calculate_entropy = entropy_coeff != 0.0

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(
                    micro_batch=data,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    entropy_requires_grad=entropy_coeff != 0.0,
                )

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio,
                                                                              cliprange_low=clip_ratio_low,
                                                                              cliprange_high=clip_ratio_high,
                                                                              loss_agg_mode=loss_agg_mode,
                                                                              normalization_factor=token_normalization)
                # compute entropy loss from entropy
                if entropy is None:
                    entropy_loss = log_prob.new_zeros(())
                else:
                    entropy_loss = core_algos.aggregate_loss(
                        loss_matrix=entropy,
                        eos_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        normalization_factor=token_normalization,
                    )

                kl_loss = None
                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = core_algos.aggregate_loss(
                        loss_matrix=kld,
                        eos_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        normalization_factor=token_normalization,
                    )

                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                policy_loss = core_algos.combine_policy_losses(
                    pg_loss=pg_loss,
                    entropy_loss=entropy_loss,
                    entropy_coeff=entropy_coeff,
                    kl_loss=kl_loss,
                    kl_loss_coef=self.config.kl_loss_coef,
                )

                # Token-mean micro-batches are already normalized against every
                # response token in the distributed PPO mini-batch.
                loss = policy_loss if loss_agg_mode == 'token-mean' else policy_loss / self.gradient_accumulation
                try:
                    loss.backward()
                except torch.OutOfMemoryError:
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    print(
                        f'actor backward OOM rank={rank} tokens={int(attention_mask.sum().item())}\n'
                        + torch.cuda.memory_summary(abbreviated=True),
                        flush=True,
                    )
                    raise

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        if self.moco_cagrad_shadow_enabled:
            observed_steps = self.moco_cagrad_tracker.step - initial_tracker_step
            if observed_steps != expected_actor_steps:
                raise RuntimeError(
                    f'MoCo--CAGrad represented {observed_steps}/{expected_actor_steps} actor steps'
                )
        self.actor_optimizer.zero_grad()
        return metrics
