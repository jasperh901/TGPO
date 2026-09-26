#!/usr/bin/env python3
"""Read-only primitives for a sharded full-vector MoCo--CAGrad shadow.

The formal LLM launchers do not import this module.  It isolates the vector
semantics that a future worker adapter must preserve before any treatment is
authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Sequence

import torch


TensorMap = Mapping[str, torch.Tensor]
ReduceSum = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrackerDiagnostics:
    pre_projection_norms: torch.Tensor
    projection_scales: torch.Tensor
    projection_flags: torch.Tensor


@dataclass(frozen=True)
class CAGradSolution:
    dual_weights: torch.Tensor
    scale: float
    effective_coefficients: torch.Tensor


def _identity_reduce(value: torch.Tensor) -> torch.Tensor:
    return value


def _stack_objective_gradients(
    objective_gradients: Sequence[TensorMap],
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if len(objective_gradients) < 2:
        raise ValueError("MoCo--CAGrad requires at least two objective gradients")
    names = tuple(sorted(objective_gradients[0]))
    if not names:
        raise ValueError("Objective gradients contain no parameter shards")
    expected = set(names)
    for objective in objective_gradients:
        if set(objective) != expected:
            raise ValueError("Every objective must contain the same parameter shards")

    stacked = {}
    for name in names:
        reference = objective_gradients[0][name]
        if not torch.is_tensor(reference):
            raise TypeError(f"Gradient shard {name!r} is not a tensor")
        tensors = []
        for objective in objective_gradients:
            tensor = objective[name]
            if tensor.shape != reference.shape or tensor.device != reference.device:
                raise ValueError(f"Gradient shard {name!r} has inconsistent shape or device")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Gradient shard {name!r} contains a non-finite value")
            tensors.append(tensor.detach().to(dtype=dtype))
        stacked[name] = torch.stack(tensors, dim=0)
    return stacked


def _validate_tracker(tracker: TensorMap) -> tuple[int, torch.device, torch.dtype]:
    if not tracker:
        raise ValueError("Tracker contains no parameter shards")
    objective_count = None
    device = None
    dtype = None
    for name, tensor in tracker.items():
        if not torch.is_tensor(tensor) or tensor.ndim < 1:
            raise ValueError(f"Tracker shard {name!r} must start with an objective axis")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Tracker shard {name!r} contains a non-finite value")
        objective_count = tensor.shape[0] if objective_count is None else objective_count
        device = tensor.device if device is None else device
        dtype = tensor.dtype if dtype is None else dtype
        if tensor.shape[0] != objective_count or tensor.device != device or tensor.dtype != dtype:
            raise ValueError("Tracker shards must share objective count, device, and dtype")
    if objective_count is None or objective_count < 2:
        raise ValueError("Tracker must contain at least two objectives")
    return objective_count, device, dtype


def update_tracker_shards(
    tracker: TensorMap | None,
    objective_gradients: Sequence[TensorMap],
    *,
    beta: float,
    maximum_norm: float = 1.0,
    state_dtype: torch.dtype = torch.float32,
    reduce_sum: ReduceSum = _identity_reduce,
) -> tuple[dict[str, torch.Tensor], TrackerDiagnostics]:
    """Update and globally project local FSDP tracker shards without mutation.

    ``reduce_sum`` must all-reduce the small per-objective squared-norm vector
    when shards are distributed.  It is the identity in single-process tests.
    """

    if not 0.0 < beta <= 1.0 or maximum_norm <= 0.0:
        raise ValueError("beta must be in (0,1] and maximum_norm must be positive")
    current = _stack_objective_gradients(objective_gradients, dtype=state_dtype)
    objective_count = len(objective_gradients)

    if tracker is not None:
        previous_count, previous_device, previous_dtype = _validate_tracker(tracker)
        first = next(iter(current.values()))
        if previous_count != objective_count:
            raise ValueError("Tracker and current gradients have different objective counts")
        if set(tracker) != set(current):
            raise ValueError("Tracker and current gradients have different parameter shards")
        if previous_device != first.device or previous_dtype != state_dtype:
            raise ValueError("Tracker device or dtype does not match the requested state")

    updated = {}
    for name, gradient in current.items():
        old = torch.zeros_like(gradient) if tracker is None else tracker[name].detach()
        if old.shape != gradient.shape:
            raise ValueError(f"Tracker shard {name!r} changed shape")
        updated[name] = (old * (1.0 - beta) + gradient * beta).clone()

    local_squared_norms = torch.zeros(
        objective_count, dtype=torch.float64, device=next(iter(updated.values())).device
    )
    for tensor in updated.values():
        local_squared_norms += tensor.to(torch.float64).reshape(objective_count, -1).square().sum(dim=1)
    global_squared_norms = reduce_sum(local_squared_norms.clone())
    if global_squared_norms.shape != (objective_count,) or not torch.isfinite(global_squared_norms).all():
        raise ValueError("Reduced tracker norms are invalid")
    if (global_squared_norms < -1e-12).any():
        raise ValueError("Reduced tracker squared norms must be non-negative")

    norms = global_squared_norms.clamp_min(0.0).sqrt()
    scales = torch.minimum(
        torch.ones_like(norms),
        torch.full_like(norms, maximum_norm) / norms.clamp_min(1e-30),
    )
    for name, tensor in updated.items():
        shape = (objective_count,) + (1,) * (tensor.ndim - 1)
        updated[name] = tensor * scales.to(device=tensor.device, dtype=tensor.dtype).reshape(shape)
    diagnostics = TrackerDiagnostics(
        pre_projection_norms=norms.detach().cpu(),
        projection_scales=scales.detach().cpu(),
        projection_flags=(norms > maximum_norm).detach().cpu(),
    )
    return updated, diagnostics


def tracker_gram(
    tracker: TensorMap,
    *,
    reduce_sum: ReduceSum = _identity_reduce,
) -> torch.Tensor:
    """Return the globally summed objective Gram for local tracker shards."""

    objective_count, device, _ = _validate_tracker(tracker)
    local = torch.zeros((objective_count, objective_count), dtype=torch.float64, device=device)
    for tensor in tracker.values():
        flat = tensor.to(torch.float64).reshape(objective_count, -1)
        local += flat @ flat.T
    gram = reduce_sum(local.clone())
    if gram.shape != local.shape or not torch.isfinite(gram).all():
        raise ValueError("Reduced tracker Gram is invalid")
    gram = 0.5 * (gram + gram.T)
    if torch.linalg.eigvalsh(gram.cpu()).min().item() < -1e-8:
        raise ValueError("Reduced tracker Gram is not positive semidefinite")
    return gram


@lru_cache(maxsize=None)
def _simplex_points(objective_count: int, resolution: int) -> tuple[tuple[float, ...], ...]:
    if resolution <= 0:
        raise ValueError("Simplex resolution must be positive")
    if objective_count == 2:
        return tuple(
            (first / resolution, 1.0 - first / resolution)
            for first in range(resolution + 1)
        )
    if objective_count == 3:
        return tuple(
            (
                first / resolution,
                second / resolution,
                (resolution - first - second) / resolution,
            )
            for first in range(resolution + 1)
            for second in range(resolution + 1 - first)
        )
    raise ValueError("The frozen shadow supports exactly two or three objectives")


def simplex_grid(objective_count: int, resolution: int = 100) -> torch.Tensor:
    return torch.tensor(_simplex_points(objective_count, resolution), dtype=torch.float64)


def cagrad_coefficients_from_gram(
    gram: torch.Tensor,
    *,
    radius: float = 0.125,
    base_coefficients: Sequence[float] | torch.Tensor | None = None,
    resolution: int = 100,
) -> CAGradSolution:
    """Solve the same deterministic CAGrad grid problem used in Amendment 042."""

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("CAGrad requires a square objective Gram")
    objective_count = gram.shape[0]
    if objective_count not in (2, 3) or not 0.0 <= radius < 1.0:
        raise ValueError("CAGrad supports K in {2,3} and radius in [0,1)")
    gram64 = gram.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(gram64).all():
        raise ValueError("CAGrad Gram contains a non-finite value")
    gram64 = 0.5 * (gram64 + gram64.T)
    if torch.linalg.eigvalsh(gram64).min().item() < -1e-8:
        raise ValueError("CAGrad Gram is not positive semidefinite")

    if base_coefficients is None:
        base = torch.full((objective_count,), 1.0 / objective_count, dtype=torch.float64)
    else:
        base = torch.as_tensor(base_coefficients, dtype=torch.float64).clone()
        if base.shape != (objective_count,) or not torch.isfinite(base).all() or (base < 0.0).any():
            raise ValueError("Base coefficients must be a finite non-negative objective vector")
        if abs(base.sum().item() - 1.0) > 1e-10:
            raise ValueError("Base coefficients must sum to one")

    base_norm = torch.sqrt(torch.clamp(base @ gram64 @ base, min=0.0)).item()
    if radius == 0.0 or base_norm <= 1e-12:
        return CAGradSolution(base, 0.0, base.clone())

    candidates = simplex_grid(objective_count, resolution)
    linear = candidates @ gram64 @ base
    candidate_norms = torch.sqrt(torch.clamp(
        torch.einsum("nk,kl,nl->n", candidates, gram64, candidates), min=0.0
    ))
    dual_objective = linear + radius * base_norm * candidate_norms
    dual = candidates[int(torch.argmin(dual_objective).item())].clone()
    weighted_norm = torch.sqrt(torch.clamp(dual @ gram64 @ dual, min=0.0)).item()
    scale = radius * base_norm / max(weighted_norm, 1e-12)
    return CAGradSolution(dual, scale, base + scale * dual)


def compose_tracker_direction(
    tracker: TensorMap,
    coefficients: Sequence[float] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Construct local direction shards directly from tracked vectors."""

    objective_count, _, _ = _validate_tracker(tracker)
    weights = torch.as_tensor(coefficients, dtype=torch.float64)
    if weights.shape != (objective_count,) or not torch.isfinite(weights).all():
        raise ValueError("Direction coefficients do not match the tracker")
    output = {}
    for name, tensor in tracker.items():
        shape = (objective_count,) + (1,) * (tensor.ndim - 1)
        local_weights = weights.to(device=tensor.device, dtype=tensor.dtype).reshape(shape)
        output[name] = (tensor * local_weights).sum(dim=0)
    return output


def cagrad_tracker_direction(
    tracker: TensorMap,
    *,
    radius: float = 0.125,
    base_coefficients: Sequence[float] | torch.Tensor | None = None,
    resolution: int = 100,
    reduce_sum: ReduceSum = _identity_reduce,
) -> tuple[dict[str, torch.Tensor], CAGradSolution, torch.Tensor]:
    gram = tracker_gram(tracker, reduce_sum=reduce_sum)
    solution = cagrad_coefficients_from_gram(
        gram,
        radius=radius,
        base_coefficients=base_coefficients,
        resolution=resolution,
    )
    direction = compose_tracker_direction(tracker, solution.effective_coefficients)
    return direction, solution, gram.detach().cpu()
