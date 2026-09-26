"""Run with torchrun --standalone --nproc-per-node=2 (CPU/Gloo)."""
import contextlib
import copy
import os
import unittest

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.checkpoint import checkpoint

from moco_cagrad_shadow import FullVectorTracker, prepare_active_update, capture_rng_state, restore_rng_state
from moco_checkpoint import tracker_state_dict, load_tracker_state_dict
from moco_fsdp_cache import FixedWeightFSDPCache


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 13)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        return torch.tanh(self.dropout(self.linear(x)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FSDP(Block(), device_id=torch.device('cpu')) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return x


class DistributedCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'RANK' not in os.environ:
            raise unittest.SkipTest('requires torchrun')
        torch.set_num_threads(1)
        torch.distributed.init_process_group('gloo')

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()

    def test_exact_gradients_tracker_rng_and_parameter_updates(self):
        torch.manual_seed(19)
        model = FSDP(Model(), device_id=torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        tracker = FullVectorTracker(3, 400, diagnostic_steps=[400], gather_reference_steps=[400])
        inputs = torch.linspace(-1, 1, 13*7).reshape(7, 13) + torch.distributed.get_rank() * 0.01

        def objective(index):
            for micro in inputs.split(3):
                (-model(micro)[:, index].mean()).backward()

        def shared():
            for micro in inputs.split(3):
                (model(micro).square().mean() * 0.03).backward()

        def reduce(value):
            torch.distributed.all_reduce(value)
            return value

        kwargs = dict(module=model, optimizer=optimizer, scheduler=None, tracker=tracker,
                      backward_objective=objective, backward_shared=shared, gradient_clip=1.0,
                      reduce_sum=reduce)
        # Start with a nonzero history and initialized FSDP handles.
        prepare_active_update(**kwargs)
        optimizer.step()
        for iteration in range(3):
            before = tracker_state_dict(tracker)
            rng = capture_rng_state()
            reference, _ = prepare_active_update(**kwargs)
            reference_gradients = [p.grad.clone() for p in model.parameters()]
            reference_rng = capture_rng_state()
            reference_tracker = tracker_state_dict(tracker)
            load_tracker_state_dict(tracker, before, model.named_parameters())
            restore_rng_state(rng)
            cache = FixedWeightFSDPCache(model, 2000)
            with cache:
                candidate, _ = prepare_active_update(**kwargs)
            self.assertGreater(cache.suppressed_frees, 0)
            self.assertFalse(cache.active)
            for param, expected in zip(model.parameters(), reference_gradients):
                self.assertTrue(torch.equal(param.grad, expected))
            self.assertTrue(torch.equal(capture_rng_state().torch_cpu, reference_rng.torch_cpu))
            self.assertEqual(reference['effective_coefficients'], candidate['effective_coefficients'])
            for key, value in reference_tracker['shards'].items():
                self.assertTrue(torch.equal(tracker.shards[key], value))
            for _, handle in cache.selected:
                self.assertEqual(handle.flat_param._full_param_padded.untyped_storage().size(), 0)
            optimizer.step()


if __name__ == '__main__':
    unittest.main(verbosity=2)
