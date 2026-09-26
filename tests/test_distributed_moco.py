"""Small multi-rank probe for the full-vector tracker reductions.

Run under ``torchrun --nproc-per-node=2``.  The ordinary unit-test discovery
skips this test unless a process group is already initialized.
"""

from __future__ import annotations

import os
import unittest

import torch

from moco_cagrad_shadow import FullVectorTracker, tracker_host_memory_preflight


class DistributedTrackerProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "RANK" in os.environ and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo")

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_variable_shards_and_both_reference_paths(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            self.skipTest("requires torchrun process group")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.assertGreaterEqual(world_size, 2)

        parameter = torch.nn.Parameter(torch.zeros(5 + rank))
        preflight = tracker_host_memory_preflight(
            torch.nn.ParameterList([parameter]),
            2,
            minimum_headroom_gib=0.0,
            allocation_overhead_factor=1.0,
            available_bytes=2**30,
        )
        self.assertEqual(preflight["global_trainable_numel"], 11)
        self.assertEqual(preflight["raw_tracker_bytes"], 11 * 2 * 4)
        self.assertEqual(preflight["status"], "PASS")
        tracker = FullVectorTracker(
            objective_count=2,
            horizon_actor_updates=16,
            maximum_norm=100.0,
            gather_reference_steps=(1,),
            gather_chunk_elements=3,
        )

        def reduce_sum(value):
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            return value

        for step in range(2):
            tracker.begin_step([("weight", parameter)])
            # Deliberately use uneven FSDP-like local shards: rank 1 owns one
            # more element than rank 0, so fixed five-element fixtures would
            # silently stop exercising the distributed path.
            element_ids = torch.arange(parameter.numel(), dtype=torch.float32)
            parameter.grad = -(element_ids + rank + 1.0)
            tracker.absorb_loss_gradients(0, [("weight", parameter)])
            parameter.grad = torch.flip(element_ids, dims=(0,)) + 0.25 * (rank + 1.0)
            tracker.absorb_loss_gradients(1, [("weight", parameter)])
            result = tracker.finalize(reduce_sum=reduce_sum)
            self.assertLessEqual(result.direct_reference_relative_error, 1e-6)
            if step == 0:
                self.assertEqual(result.reference_mode, "chunked_all_gather_full_direction")
            else:
                self.assertEqual(result.reference_mode, "sharded_fp64_global_l2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
