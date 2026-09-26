"""Cheap, model-free refresh policy for amortized MoCo--CAGrad geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RefreshDecision:
    due: bool
    reason: str
    rare_fraction: float
    drift: float


def objective_signature(
    objective_advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a small reward-space signature without building autograd graphs.

    The signature contains per-objective means and the normalized covariance
    matrix.  It is only a trigger proxy; the actual CAGrad Gram matrix remains
    the parameter-space quantity computed during a refresh.
    """

    if objective_advantages.ndim != 3:
        raise ValueError("objective_advantages must have shape [batch, objective, response]")
    if response_mask.shape != objective_advantages.shape[::2]:
        raise ValueError("response_mask must have shape [batch, response]")
    values = objective_advantages.detach().to(dtype=torch.float32)
    mask = response_mask.detach().to(device=values.device, dtype=torch.float32)
    count = mask.sum().clamp_min(1.0)
    flat = values * mask[:, None, :]
    means = flat.sum(dim=(0, 2)) / count
    centered = values - means[None, :, None]
    centered = centered * mask[:, None, :]
    matrix = centered.reshape(centered.shape[1], -1)
    covariance = (matrix @ matrix.T) / count
    scale = covariance.diag().clamp_min(1.0e-8).sqrt()
    normalized = covariance / (scale[:, None] * scale[None, :]).clamp_min(1.0e-8)
    return torch.cat((means, normalized.reshape(-1))).detach().cpu()


def signature_drift(current: torch.Tensor, reference: torch.Tensor | None) -> float:
    if reference is None:
        # The first step is independently forced to refresh; keep the value
        # finite because actor audit records use strict JSON serialization.
        return 0.0
    current = current.detach().to(dtype=torch.float64)
    reference = reference.detach().to(dtype=torch.float64)
    if current.shape != reference.shape:
        raise ValueError("Current and reference signatures have different shapes")
    denominator = torch.linalg.vector_norm(reference).clamp_min(1.0e-6)
    return float((torch.linalg.vector_norm(current - reference) / denominator).item())


def decide_refresh(
    *,
    first_step: bool,
    scheduled: bool,
    rare_fraction: float,
    drift: float,
    adaptive_enabled: bool,
    rare_fraction_threshold: float,
    drift_threshold: float,
) -> RefreshDecision:
    if not 0.0 <= rare_fraction <= 1.0:
        raise ValueError("rare_fraction must lie in [0, 1]")
    if rare_fraction_threshold < 0.0 or rare_fraction_threshold > 1.0:
        raise ValueError("rare_fraction_threshold must lie in [0, 1]")
    if drift_threshold < 0.0:
        raise ValueError("drift_threshold must be non-negative")
    if first_step:
        return RefreshDecision(True, "initial", rare_fraction, drift)
    if scheduled:
        return RefreshDecision(True, "scheduled", rare_fraction, drift)
    if not adaptive_enabled:
        return RefreshDecision(False, "stale_interval", rare_fraction, drift)
    if rare_fraction >= rare_fraction_threshold:
        return RefreshDecision(True, "rare_target", rare_fraction, drift)
    if drift >= drift_threshold:
        return RefreshDecision(True, "advantage_drift", rare_fraction, drift)
    return RefreshDecision(False, "adaptive_reuse", rare_fraction, drift)
