import copy
import importlib.util
import math
import os
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from moco_cagrad_shadow import (  # noqa: E402
    FullVectorTracker,
    capture_rng_state,
    execute_readonly_shadow_step,
    prepare_active_update,
    rng_hash,
    tracker_host_memory_preflight,
    training_state_hashes,
)


REFERENCE_PATH = Path(os.environ.get('MOCO_VECTOR_REFERENCE', str(
    Path(__file__).resolve().parent / "references"
    / "moco_cagrad_vector_core.py"
)))
REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "frozen_moco_cagrad_vector_reference", REFERENCE_PATH
)
assert REFERENCE_SPEC is not None and REFERENCE_SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
sys.modules[REFERENCE_SPEC.name] = REFERENCE
REFERENCE_SPEC.loader.exec_module(REFERENCE)


class TinyDropoutActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 7)
        self.dropout = torch.nn.Dropout(0.4)
        self.second = torch.nn.Linear(7, 2)

    def forward(self, inputs):
        return self.second(torch.tanh(self.dropout(self.first(inputs))))


def make_tiny(seed=11):
    torch.manual_seed(seed)
    model = TinyDropoutActor()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def make_callbacks(model, inputs):
    def backward_objective(objective_id):
        # A minimized negative score produces a reward-ascent gradient.
        (-model(inputs)[:, objective_id].mean()).backward()

    def backward_shared():
        # Positive minimized penalty; the shadow converts it to ascent sign.
        (0.03 * model(inputs).square().mean()).backward()

    return backward_objective, backward_shared


def baseline_step(model, optimizer, scheduler, inputs):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = -(0.55 * outputs[:, 0] + 0.45 * outputs[:, 1]).mean()
    loss = loss + 0.03 * outputs.square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def assert_nested_equal(testcase, first, second):
    if torch.is_tensor(first):
        testcase.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        testcase.assertEqual(set(first), set(second))
        for key in first:
            assert_nested_equal(testcase, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        testcase.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            assert_nested_equal(testcase, left, right)
    else:
        testcase.assertEqual(first, second)


class TestFullVectorShadow(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(101)
        np.random.seed(101)
        random.seed(101)

    def test_readonly_step_restores_all_state_and_rng(self):
        model, optimizer, scheduler = make_tiny()
        inputs = torch.randn(9, 4)
        objective, shared = make_callbacks(model, inputs)
        tracker = FullVectorTracker(2, 16, maximum_norm=1.0)
        before = training_state_hashes(model, optimizer, scheduler)
        before_rng = rng_hash(capture_rng_state())

        record, _ = execute_readonly_shadow_step(
            module=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tracker=tracker,
            backward_objective=objective,
            backward_shared=shared,
            gradient_clip=1.0,
        )

        self.assertEqual(before, training_state_hashes(model, optimizer, scheduler))
        self.assertEqual(before_rng, rng_hash(capture_rng_state()))
        self.assertTrue(record["state_unchanged"])
        self.assertEqual(record["tracker_step"], 1)
        self.assertEqual(record["horizon_actor_updates"], 16)
        self.assertAlmostEqual(record["beta"], 0.25)
        self.assertLessEqual(record["direct_reference_relative_error"], 1e-6)
        self.assertTrue(math.isfinite(record["bf16_direction_cosine"]))
        self.assertEqual(record["state_hashes_before"], record["state_hashes_after"])

    def test_shadow_then_baseline_matches_uninterrupted_dropout_update(self):
        first_model, first_optimizer, first_scheduler = make_tiny(seed=29)
        second_model, second_optimizer, second_scheduler = make_tiny(seed=29)
        second_model.load_state_dict(copy.deepcopy(first_model.state_dict()))
        second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))
        second_scheduler.load_state_dict(copy.deepcopy(first_scheduler.state_dict()))
        inputs = torch.linspace(-1.2, 1.1, 36).reshape(9, 4)
        objective, shared = make_callbacks(first_model, inputs)
        tracker = FullVectorTracker(2, 25, maximum_norm=1.0)

        torch.manual_seed(777)
        np.random.seed(777)
        random.seed(777)
        execute_readonly_shadow_step(
            module=first_model,
            optimizer=first_optimizer,
            scheduler=first_scheduler,
            tracker=tracker,
            backward_objective=objective,
            backward_shared=shared,
            gradient_clip=1.0,
        )
        baseline_step(first_model, first_optimizer, first_scheduler, inputs)

        torch.manual_seed(777)
        np.random.seed(777)
        random.seed(777)
        baseline_step(second_model, second_optimizer, second_scheduler, inputs)

        for first, second in zip(first_model.parameters(), second_model.parameters()):
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        assert_nested_equal(self, first_optimizer.state_dict(), second_optimizer.state_dict())
        assert_nested_equal(self, first_scheduler.state_dict(), second_scheduler.state_dict())

    def test_adamw_preview_matches_real_torch_step_with_history(self):
        torch.manual_seed(41)
        model = torch.nn.Linear(3, 1, bias=False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-3, betas=(0.8, 0.97), eps=2e-7, weight_decay=0.03
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        parameter = model.weight

        # Initialize nonzero AdamW history before the read-only candidate step.
        parameter.grad = torch.tensor([[0.04, -0.02, 0.01]])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        inputs = torch.tensor([[1.0, -0.5, 0.2], [-0.4, 0.7, 1.1]])

        def objective(objective_id):
            output = model(inputs).squeeze(-1)
            if objective_id == 0:
                (-output.mean()).backward()
            else:
                (-(output * torch.tensor([0.3, 1.2])).mean()).backward()

        tracker = FullVectorTracker(2, 1, maximum_norm=100.0)
        before = parameter.detach().clone()
        record, displacements = execute_readonly_shadow_step(
            module=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tracker=tracker,
            backward_objective=objective,
            backward_shared=None,
            gradient_clip=1.0,
            return_displacements=True,
            adamw_preview_chunk_elements=2,
        )
        self.assertIsNotNone(displacements)
        expected = displacements["weight"].reshape_as(parameter)

        candidate_ascent = tracker.direction_shard("weight").reshape_as(parameter)
        parameter.grad = -candidate_ascent
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        observed = parameter.detach() - before
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-8)
        self.assertEqual(record["adamw_next_steps"], [2])
        self.assertEqual(record["adamw_preview_chunk_elements"], 2)

    def test_active_update_loads_one_combined_gradient_and_matches_preview(self):
        torch.manual_seed(83)
        model = torch.nn.Linear(3, 2, bias=False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-3, betas=(0.8, 0.97), eps=2e-7,
            weight_decay=0.03,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        inputs = torch.tensor([[1.0, -0.5, 0.2], [-0.4, 0.7, 1.1]])

        def objective(objective_id):
            (-model(inputs)[:, objective_id].mean()).backward()

        def shared():
            (0.02 * model(inputs).square().mean()).backward()

        tracker = FullVectorTracker(
            2, 1, maximum_norm=100.0, radius=0.125,
            diagnostic_steps=(1,),
        )
        before = model.weight.detach().clone()
        before_state = training_state_hashes(model, optimizer, scheduler)
        record, displacements = prepare_active_update(
            module=model,
            optimizer=optimizer,
            scheduler=scheduler,
            tracker=tracker,
            backward_objective=objective,
            backward_shared=shared,
            gradient_clip=1.0,
            preview_adamw=True,
            return_displacements=True,
            adamw_preview_chunk_elements=2,
        )

        self.assertEqual(before_state, training_state_hashes(model, optimizer, scheduler))
        self.assertEqual(record["update_mode"], "active")
        self.assertTrue(record["gradient_loaded"])
        self.assertFalse(record["optimizer_step_applied"])
        self.assertTrue(record["state_unchanged_before_optimizer"])
        self.assertTrue(record["diagnostics_audited"])
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(displacements)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        observed = model.weight.detach() - before
        expected = displacements["weight"].reshape_as(model.weight)
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-8)

    def test_active_diagnostic_cadence_skips_only_optional_audits(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        named = [("weight", parameter)]
        tracker = FullVectorTracker(
            2, 3, maximum_norm=100.0,
            gather_reference_steps=(1, 3),
            diagnostic_steps=(1, 3),
        )
        for expected_step in range(1, 4):
            tracker.begin_step(named)
            parameter.grad = torch.tensor([-1.0, 0.0])
            tracker.absorb_loss_gradients(0, named)
            parameter.grad = torch.tensor([0.0, -1.0])
            tracker.absorb_loss_gradients(1, named)
            observed = tracker.finalize()
            self.assertEqual(observed.step, expected_step)
            self.assertEqual(observed.diagnostics_audited, expected_step in (1, 3))
            if expected_step == 2:
                self.assertEqual(observed.reference_mode, "not_scheduled")
                self.assertEqual(observed.direct_reference_relative_error, -1.0)
            self.assertTrue(torch.isfinite(observed.gram).all())

    def test_tracker_recursion_uses_actor_update_horizon(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        tracker = FullVectorTracker(2, 4, maximum_norm=100.0)
        named = [("weight", parameter)]

        tracker.begin_step(named)
        parameter.grad = torch.tensor([-2.0, 0.0])
        tracker.absorb_loss_gradients(0, named)
        parameter.grad = torch.tensor([0.0, -4.0])
        tracker.absorb_loss_gradients(1, named)
        tracker.finalize()
        torch.testing.assert_close(
            tracker.shards["weight"], torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        )

        tracker.begin_step(named)
        parameter.grad = torch.tensor([-4.0, 0.0])
        tracker.absorb_loss_gradients(0, named)
        parameter.grad = torch.tensor([0.0, -2.0])
        tracker.absorb_loss_gradients(1, named)
        tracker.finalize()
        torch.testing.assert_close(
            tracker.shards["weight"], torch.tensor([[2.5, 0.0], [0.0, 2.0]])
        )

    def test_multistep_tracker_matches_independent_frozen_reference(self):
        generator = torch.Generator().manual_seed(2026082701)
        for objective_count, horizon in ((2, 17), (3, 29)):
            parameters = [
                ("left", torch.nn.Parameter(torch.zeros(2, 3))),
                ("right", torch.nn.Parameter(torch.zeros(5))),
            ]
            tracker = FullVectorTracker(
                objective_count,
                horizon,
                maximum_norm=0.7,
                radius=0.125,
                resolution=100,
            )
            reference_tracker = None
            for _ in range(5):
                objective_gradients = []
                tracker.begin_step(parameters)
                for objective_id in range(objective_count):
                    gradients = {
                        name: torch.randn(
                            parameter.shape, generator=generator, dtype=torch.float32
                        )
                        for name, parameter in parameters
                    }
                    objective_gradients.append(gradients)
                    for name, parameter in parameters:
                        parameter.grad = -gradients[name]
                    tracker.absorb_loss_gradients(objective_id, parameters)
                observed = tracker.finalize()

                reference_tracker, reference_projection = REFERENCE.update_tracker_shards(
                    reference_tracker,
                    objective_gradients,
                    beta=1.0 / math.sqrt(horizon),
                    maximum_norm=0.7,
                )
                reference_direction, reference_solution, reference_gram = (
                    REFERENCE.cagrad_tracker_direction(
                        reference_tracker, radius=0.125, resolution=100
                    )
                )

                for name, _ in parameters:
                    torch.testing.assert_close(
                        tracker.shards[name],
                        reference_tracker[name].reshape(objective_count, -1),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                    torch.testing.assert_close(
                        tracker.direction_shard(name),
                        reference_direction[name].reshape(-1),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                torch.testing.assert_close(
                    observed.projection.pre_projection_norms,
                    reference_projection.pre_projection_norms,
                )
                torch.testing.assert_close(observed.gram, reference_gram)
                torch.testing.assert_close(
                    observed.solution.dual_weights,
                    reference_solution.dual_weights,
                )
                torch.testing.assert_close(
                    observed.solution.effective_coefficients,
                    reference_solution.effective_coefficients,
                )

    def test_training_state_mutation_fails_closed(self):
        model, optimizer, scheduler = make_tiny(seed=53)
        inputs = torch.randn(5, 4)
        tracker = FullVectorTracker(2, 4)

        def mutating_objective(objective_id):
            if objective_id == 0:
                with torch.no_grad():
                    next(model.parameters()).add_(0.5)
            (-model(inputs)[:, objective_id].mean()).backward()

        with self.assertRaisesRegex(RuntimeError, "modified training state"):
            execute_readonly_shadow_step(
                module=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tracker=tracker,
                backward_objective=mutating_objective,
                backward_shared=None,
                gradient_clip=1.0,
            )
        self.assertTrue(tracker.poisoned)

    def test_host_memory_preflight_uses_objective_count_and_margin(self):
        model = torch.nn.Linear(4, 2, bias=False)
        report = tracker_host_memory_preflight(
            model,
            3,
            minimum_headroom_gib=1.0,
            allocation_overhead_factor=1.10,
            available_bytes=2**30 + 2000,
        )
        self.assertEqual(report["global_trainable_numel"], 8)
        self.assertEqual(report["raw_tracker_bytes"], 8 * 3 * 4)
        self.assertEqual(report["future_optimizer_bytes"], 0)
        self.assertEqual(report["status"], "PASS")

        with self.assertRaisesRegex(MemoryError, "Insufficient host memory"):
            tracker_host_memory_preflight(
                model,
                3,
                minimum_headroom_gib=1.0,
                allocation_overhead_factor=1.10,
                available_bytes=2**30,
            )

    def test_host_memory_preflight_counts_lazy_offloaded_adamw_state(self):
        model = torch.nn.Linear(4, 2, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        report = tracker_host_memory_preflight(
            model,
            3,
            optimizer=optimizer,
            optimizer_offload=True,
            minimum_headroom_gib=0.0,
            allocation_overhead_factor=1.0,
            available_bytes=10_000,
        )
        self.assertEqual(report["raw_tracker_bytes"], 8 * 3 * 4)
        self.assertEqual(report["future_optimizer_bytes"], 8 * 2 * 4)
        self.assertEqual(report["raw_future_bytes"], 8 * 5 * 4)

        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = state[key].cpu()
        initialized = tracker_host_memory_preflight(
            model,
            3,
            optimizer=optimizer,
            optimizer_offload=True,
            minimum_headroom_gib=0.0,
            allocation_overhead_factor=1.0,
            available_bytes=10_000,
        )
        self.assertEqual(initialized["future_optimizer_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
