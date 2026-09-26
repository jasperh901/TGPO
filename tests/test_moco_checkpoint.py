import copy
import io
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from moco_cagrad_shadow import FullVectorTracker
from moco_checkpoint import tracker_state_dict, load_tracker_state_dict
from moco_checkpoint import save_worker_tracker, restore_worker_tracker


class MoCoCheckpointTest(unittest.TestCase):
    def make(self):
        return FullVectorTracker(3, 400, diagnostic_steps=[400], gather_reference_steps=[400],
                                 gather_chunk_elements=13)

    def advance(self, tracker, parameters, gradients):
        tracker.begin_step(parameters)
        for index in range(3):
            for name, parameter in parameters:
                parameter.grad = gradients[index][name].clone()
            tracker.absorb_loss_gradients(index, parameters)
        return tracker.finalize()

    def test_restored_history_matches_uninterrupted_exactly(self):
        torch.manual_seed(218)
        params = [('a', torch.nn.Parameter(torch.randn(71))),
                  ('b', torch.nn.Parameter(torch.randn(11, 3)))]
        gradients = [[{name: torch.randn_like(p) for name, p in params} for _ in range(3)]
                     for _ in range(6)]
        old = self.make()
        for grad in gradients[:3]:
            self.advance(old, params, grad)
        saved = tracker_state_dict(old)
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        restored = self.make()
        load_tracker_state_dict(restored, torch.load(stream, weights_only=False), params)
        for grad in gradients[3:]:
            first = self.advance(old, params, grad)
            second = self.advance(restored, params, grad)
            self.assertEqual(first.step, second.step)
            self.assertTrue(torch.equal(first.gram, second.gram))
            self.assertTrue(torch.equal(first.solution.effective_coefficients,
                                        second.solution.effective_coefficients))
            for name, _ in params:
                self.assertTrue(torch.equal(old.shards[name], restored.shards[name]))
                self.assertTrue(torch.equal(old.direction_shard(name), restored.direction_shard(name)))

    def test_fail_closed_and_detached_snapshot(self):
        parameter = torch.nn.Parameter(torch.ones(31))
        params = [('a', parameter)]
        tracker = self.make()
        self.advance(tracker, params, [{'a': torch.ones(31)} for _ in range(3)])
        saved = tracker_state_dict(tracker)
        tracker.shards['a'].zero_()
        self.assertGreater(saved['shards']['a'].abs().sum().item(), 0)
        bad = copy.deepcopy(saved)
        bad['config']['radius'] = 0.5
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            load_tracker_state_dict(self.make(), bad, params)
        with self.assertRaisesRegex(ValueError, 'ownership differs'):
            load_tracker_state_dict(self.make(), saved, [('other', parameter)])
        tracker.begin_step(params)
        with self.assertRaisesRegex(RuntimeError, 'unfinished'):
            tracker_state_dict(tracker)

    def test_worker_snapshot_requires_matching_rank_and_history(self):
        module = torch.nn.Linear(3, 2)
        parameters = list(module.named_parameters())
        tracker = self.make()
        gradients = [{name: torch.randn_like(p) for name, p in parameters} for _ in range(3)]
        self.advance(tracker, parameters, gradients)
        worker = SimpleNamespace(rank=0, actor=SimpleNamespace(
            moco_cagrad_tracker=tracker, actor_module=module))
        with tempfile.TemporaryDirectory() as tmp, patch('torch.distributed.get_world_size', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'Missing MoCo history'):
                restore_worker_tracker(worker, tmp)
            save_worker_tracker(worker, tmp)
            worker.actor.moco_cagrad_tracker = self.make()
            restore_worker_tracker(worker, tmp)
            for name, _ in parameters:
                self.assertTrue(torch.equal(tracker.shards[name], worker.actor.moco_cagrad_tracker.shards[name]))
            with patch('torch.distributed.get_world_size', return_value=4):
                with self.assertRaisesRegex(ValueError, 'rank layout differs'):
                    restore_worker_tracker(worker, tmp)


if __name__ == '__main__':
    unittest.main()
