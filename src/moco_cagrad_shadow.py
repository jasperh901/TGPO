"""Full-vector MoCo--CAGrad mechanics for sharded PPO actors.

The canonical tracker representation is sharded CPU FP32.  The read-only path
proves that candidate mechanics cannot mutate a baseline update; the active
path prepares one minimizing gradient for the caller's ordinary FSDP/AdamW
step.  Optimizer and scheduler ownership therefore remains with VERL.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch


ReduceSum = Callable[[torch.Tensor], torch.Tensor]


def _identity_reduce(value: torch.Tensor) -> torch.Tensor:
    return value


def _host_mem_available_bytes() -> int:
    """Read the kernel's reclaimable host-memory estimate."""

    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 3 and fields[0] == "MemAvailable:":
                    if fields[2] != "kB":
                        raise RuntimeError("/proc/meminfo MemAvailable is not in kB")
                    return int(fields[1]) * 1024
    except (OSError, ValueError) as error:
        raise RuntimeError("Unable to read /proc/meminfo MemAvailable") from error
    raise RuntimeError("/proc/meminfo has no MemAvailable entry")


def tracker_host_memory_preflight(
    module: torch.nn.Module,
    objective_count: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_offload: bool = False,
    minimum_headroom_gib: float = 8.0,
    allocation_overhead_factor: float = 1.10,
    available_bytes: int | None = None,
) -> dict:
    """Bound CPU tracker allocation before the first shadow step.

    FSDP exposes a local parameter shard on each rank.  The all-reduced count
    therefore measures the actual aggregate tracker allocation, including any
    accidental replication caused by a parallelism layout.  The check is
    deliberately fail-closed and can be supplied a fixed ``available_bytes``
    value by unit tests.
    """

    if objective_count not in (2, 3):
        raise ValueError("Tracker host preflight supports two or three objectives")
    if minimum_headroom_gib < 0.0:
        raise ValueError("Minimum host headroom must be non-negative")
    if allocation_overhead_factor < 1.0:
        raise ValueError("Allocation overhead factor must be at least one")

    trainable_parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    local_numel = sum(int(parameter.numel()) for parameter in trainable_parameters)
    local_future_optimizer_bytes = 0
    if optimizer_offload:
        if optimizer is None:
            raise ValueError("Optimizer state estimation requires an optimizer")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("Optimizer state estimation requires AdamW")
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_parameters != {id(parameter) for parameter in trainable_parameters}:
            raise ValueError("Optimizer and actor trainable parameter layouts differ")
        for parameter in trainable_parameters:
            state = optimizer.state.get(parameter, {})
            # AdamW creates FP32 exp_avg and exp_avg_sq lazily at the first
            # optimizer step.  Count whichever portion is not already on CPU.
            target_bytes = 2 * int(parameter.numel()) * 4
            cpu_state_bytes = sum(
                int(value.numel()) * int(value.element_size())
                for key, value in state.items()
                if key in ("exp_avg", "exp_avg_sq")
                and torch.is_tensor(value)
                and value.device.type == "cpu"
            )
            local_future_optimizer_bytes += max(
                0, target_bytes - min(target_bytes, cpu_state_bytes)
            )
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        backend = str(torch.distributed.get_backend()).lower()
        reduction_device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl" else torch.device("cpu")
        )
        counts = torch.tensor(
            [local_numel, local_future_optimizer_bytes],
            dtype=torch.int64,
            device=reduction_device,
        )
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        global_numel = int(counts[0].item())
        future_optimizer_bytes = int(counts[1].item())
    else:
        reduction_device = torch.device("cpu")
        global_numel = local_numel
        future_optimizer_bytes = local_future_optimizer_bytes

    raw_tracker_bytes = global_numel * objective_count * torch.tensor([], dtype=torch.float32).element_size()
    raw_future_bytes = raw_tracker_bytes + future_optimizer_bytes
    estimated_future_bytes = math.ceil(raw_future_bytes * allocation_overhead_factor)
    observed_available_bytes = (
        _host_mem_available_bytes() if available_bytes is None else int(available_bytes)
    )
    if observed_available_bytes < 0:
        raise ValueError("Available host memory must be non-negative")
    if distributed:
        available = torch.tensor(
            [observed_available_bytes], dtype=torch.int64, device=reduction_device
        )
        # Use the minimum observation so every rank makes the same decision.
        torch.distributed.all_reduce(available, op=torch.distributed.ReduceOp.MIN)
        observed_available_bytes = int(available.item())

    minimum_headroom_bytes = math.ceil(minimum_headroom_gib * 2**30)
    headroom_bytes = observed_available_bytes - estimated_future_bytes
    result = {
        "status": "PASS" if headroom_bytes >= minimum_headroom_bytes else "FAIL",
        "local_trainable_numel": local_numel,
        "global_trainable_numel": global_numel,
        "objective_count": objective_count,
        "raw_tracker_bytes": raw_tracker_bytes,
        "future_optimizer_bytes": future_optimizer_bytes,
        "raw_future_bytes": raw_future_bytes,
        "estimated_future_bytes": estimated_future_bytes,
        "allocation_overhead_factor": allocation_overhead_factor,
        "available_bytes": observed_available_bytes,
        "headroom_bytes": headroom_bytes,
        "minimum_headroom_bytes": minimum_headroom_bytes,
        "raw_tracker_gib": raw_tracker_bytes / 2**30,
        "future_optimizer_gib": future_optimizer_bytes / 2**30,
        "estimated_future_gib": estimated_future_bytes / 2**30,
        "available_gib": observed_available_bytes / 2**30,
        "headroom_gib": headroom_bytes / 2**30,
        "minimum_headroom_gib": minimum_headroom_gib,
    }
    if result["status"] != "PASS":
        raise MemoryError(
            "Insufficient host memory for the CPU tracker and future optimizer state: "
            f"estimated={result['estimated_future_gib']:.2f} GiB, "
            f"available={result['available_gib']:.2f} GiB, "
            f"required_headroom={minimum_headroom_gib:.2f} GiB"
        )
    return result


@dataclass(frozen=True)
class CAGradSolution:
    dual_weights: torch.Tensor
    scale: float
    effective_coefficients: torch.Tensor


@dataclass(frozen=True)
class ProjectionDiagnostics:
    pre_projection_norms: torch.Tensor
    scales: torch.Tensor
    flags: torch.Tensor


@dataclass(frozen=True)
class TrackerStep:
    step: int
    gram: torch.Tensor
    solution: CAGradSolution
    projection: ProjectionDiagnostics
    diagnostics_audited: bool
    reference_mode: str
    direct_reference_relative_error: float
    bf16_gram_relative_error: float
    bf16_direction_cosine: float
    bf16_dual_l1_error: float


@dataclass
class RngSnapshot:
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]
    numpy_state: tuple
    python_state: tuple


def capture_rng_state() -> RngSnapshot:
    # One Ray/FSDP worker owns one current CUDA stream. Touching every visible
    # device here would create needless contexts in all eight worker processes.
    cuda_states = [torch.cuda.get_rng_state().clone()] if torch.cuda.is_available() else []
    return RngSnapshot(
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=[state.clone() for state in cuda_states],
        numpy_state=np.random.get_state(),
        python_state=random.getstate(),
    )


def restore_rng_state(snapshot: RngSnapshot) -> None:
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda:
        torch.cuda.set_rng_state(snapshot.torch_cuda[0])
    np.random.set_state(snapshot.numpy_state)
    random.setstate(snapshot.python_state)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def rng_hash(snapshot: RngSnapshot) -> str:
    digest = hashlib.sha256()
    digest.update(_tensor_bytes(snapshot.torch_cpu))
    for state in snapshot.torch_cuda:
        digest.update(_tensor_bytes(state))
    digest.update(pickle.dumps(snapshot.numpy_state, protocol=5))
    digest.update(pickle.dumps(snapshot.python_state, protocol=5))
    return digest.hexdigest()


def _finite_tensor_signature(tensor: torch.Tensor) -> dict:
    """Return a constant-size mutation fingerprint for a potentially huge shard.

    Standard PyTorch in-place writes increment ``_version``. Fixed-position
    content samples additionally catch unversioned storage writes without an
    O(parameter-count) audit per shadow. Storage pointers are deliberately not
    semantic because FSDP may rebind a flat parameter while resharding.
    """

    value = tensor.detach()
    flat = value.reshape(-1)
    if flat.numel() == 0:
        sample_hash = hashlib.sha256(b"").hexdigest()
    else:
        sample_count = min(flat.numel(), 257)
        positions = torch.linspace(
            0, flat.numel() - 1, sample_count, device=flat.device, dtype=torch.float64
        ).round().to(torch.long)
        sample = flat.index_select(0, positions).to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(sample).all():
            raise ValueError("Training state sample contains a non-finite value")
        sample_hash = hashlib.sha256(_tensor_bytes(sample)).hexdigest()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": value.numel(),
        "version": int(value._version),
        "sample_hash": sample_hash,
    }


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_state_hashes(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> dict[str, str]:
    """Hash versioned, sampled state signatures without full tensor copies."""

    named_parameters = list(module.named_parameters())
    parameter_names = {id(parameter): name for name, parameter in named_parameters}
    model_signature = {
        "parameters": [
            (name, _finite_tensor_signature(parameter))
            for name, parameter in named_parameters
        ],
        "buffers": [
            (name, _finite_tensor_signature(buffer))
            for name, buffer in module.named_buffers()
        ],
    }
    optimizer_state = []
    for parameter, state in optimizer.state.items():
        name = parameter_names.get(id(parameter))
        if name is None:
            raise ValueError("Optimizer contains a parameter absent from the actor")
        items = []
        for key, value in sorted(state.items()):
            if torch.is_tensor(value):
                items.append((str(key), _finite_tensor_signature(value)))
            else:
                items.append((str(key), repr(value)))
        optimizer_state.append((name, items))
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            key: ([parameter_names[id(item)] for item in value] if key == "params" else repr(value))
            for key, value in sorted(group.items())
        })
    optimizer_signature = {
        "state": sorted(optimizer_state),
        "groups": groups,
    }
    scheduler_signature = None if scheduler is None else scheduler.state_dict()
    hashes = {
        "model": _json_hash(model_signature),
        "optimizer": _json_hash(optimizer_signature),
        "scheduler": _json_hash(scheduler_signature),
    }
    hashes["checkpoint_visible"] = hashes["model"]
    hashes["combined"] = _json_hash(hashes)
    return hashes


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
            (first / resolution, second / resolution, (resolution - first - second) / resolution)
            for first in range(resolution + 1)
            for second in range(resolution + 1 - first)
        )
    raise ValueError("The frozen solver supports exactly two or three objectives")


def cagrad_coefficients_from_gram(
    gram: torch.Tensor,
    *,
    radius: float = 0.125,
    resolution: int = 100,
) -> CAGradSolution:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("CAGrad requires a square Gram matrix")
    objective_count = gram.shape[0]
    if objective_count not in (2, 3) or not 0.0 <= radius < 1.0:
        raise ValueError("CAGrad supports K in {2,3} and radius in [0,1)")
    matrix = gram.detach().to(device="cpu", dtype=torch.float64)
    matrix = 0.5 * (matrix + matrix.T)
    if not torch.isfinite(matrix).all():
        raise ValueError("CAGrad Gram contains a non-finite value")
    if torch.linalg.eigvalsh(matrix).min().item() < -1e-8:
        raise ValueError("CAGrad Gram is not positive semidefinite")

    base = torch.full((objective_count,), 1.0 / objective_count, dtype=torch.float64)
    base_norm = torch.sqrt(torch.clamp(base @ matrix @ base, min=0.0)).item()
    if radius == 0.0 or base_norm <= 1e-12:
        return CAGradSolution(base.clone(), 0.0, base.clone())

    candidates = torch.tensor(
        _simplex_points(objective_count, resolution), dtype=torch.float64
    )
    linear = candidates @ matrix @ base
    candidate_norms = torch.sqrt(torch.clamp(
        torch.einsum("nk,kl,nl->n", candidates, matrix, candidates), min=0.0
    ))
    dual_objective = linear + radius * base_norm * candidate_norms
    dual = candidates[int(torch.argmin(dual_objective).item())].clone()
    dual_norm = torch.sqrt(torch.clamp(dual @ matrix @ dual, min=0.0)).item()
    scale = radius * base_norm / max(dual_norm, 1e-12)
    return CAGradSolution(dual, scale, base + scale * dual)


def _validate_reduced(value: torch.Tensor, expected_shape: tuple[int, ...]) -> torch.Tensor:
    if value.shape != expected_shape or not torch.isfinite(value).all():
        raise ValueError("Distributed reduction returned an invalid tensor")
    return value.detach().to(device="cpu", dtype=torch.float64)


class FullVectorTracker:
    """Persistent sharded CPU FP32 MoCo state."""

    def __init__(
        self,
        objective_count: int,
        horizon_actor_updates: int,
        *,
        maximum_norm: float = 1.0,
        radius: float = 0.125,
        resolution: int = 100,
        gather_reference_steps: Sequence[int] | None = None,
        diagnostic_steps: Sequence[int] | None = None,
        gather_chunk_elements: int = 1_048_576,
    ):
        if objective_count not in (2, 3):
            raise ValueError("Full-vector shadow requires two or three objectives")
        if horizon_actor_updates < 1 or maximum_norm <= 0.0:
            raise ValueError("Tracker horizon and norm bound must be positive")
        if gather_chunk_elements < 1:
            raise ValueError("Gather chunk size must be positive")
        if gather_reference_steps is not None:
            normalized_steps = tuple(sorted({int(step) for step in gather_reference_steps}))
            if not normalized_steps or normalized_steps[0] < 1:
                raise ValueError("Registered gather-reference steps must be positive")
        else:
            normalized_steps = None
        if diagnostic_steps is not None:
            normalized_diagnostic_steps = tuple(
                sorted({int(step) for step in diagnostic_steps})
            )
            if (not normalized_diagnostic_steps
                    or normalized_diagnostic_steps[0] < 1):
                raise ValueError("Registered diagnostic steps must be positive")
        else:
            normalized_diagnostic_steps = None
        self.objective_count = objective_count
        self.horizon_actor_updates = horizon_actor_updates
        self.beta = 1.0 / math.sqrt(horizon_actor_updates)
        self.maximum_norm = maximum_norm
        self.radius = radius
        self.resolution = resolution
        self.gather_reference_steps = normalized_steps
        self.diagnostic_steps = normalized_diagnostic_steps
        self.gather_chunk_elements = gather_chunk_elements
        self.shards: dict[str, torch.Tensor] = {}
        self.parameter_shapes: dict[str, tuple[int, ...]] = {}
        self.step = 0
        self._next_objective = 0
        self._active = False
        self.poisoned = False
        self.last_step: TrackerStep | None = None
        self.host_transfer_bytes = 0

    def begin_step(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        if self.poisoned:
            raise RuntimeError("A failed tracker cannot be reused")
        if self._active:
            raise RuntimeError("Tracker step is already active")
        observed = {
            name: tuple(parameter.shape)
            for name, parameter in named_parameters if parameter.requires_grad
        }
        if not observed:
            raise ValueError("Actor exposes no trainable parameter shards")
        if self.parameter_shapes and observed != self.parameter_shapes:
            raise RuntimeError("FSDP parameter shard layout changed between steps")
        self.parameter_shapes = observed
        self._next_objective = 0
        self._active = True

    def absorb_loss_gradients(
        self,
        objective_id: int,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> None:
        """Apply one minimized-loss gradient as a reward-ascent MoCo update."""

        if not self._active or objective_id != self._next_objective:
            raise RuntimeError("Objective gradients must be absorbed once in registered order")
        seen = set()
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            seen.add(name)
            if tuple(parameter.shape) != self.parameter_shapes[name]:
                raise RuntimeError(f"Parameter shard {name!r} changed shape")
            if name not in self.shards:
                self.shards[name] = torch.zeros(
                    (self.objective_count, parameter.numel()), dtype=torch.float32
                )
            tracker = self.shards[name]
            gradient = parameter.grad
            if gradient is None:
                tracker[objective_id].mul_(1.0 - self.beta)
            else:
                gradient_flat = gradient.detach().reshape(-1)
                for start in range(0, parameter.numel(), self.gather_chunk_elements):
                    end = min(start + self.gather_chunk_elements, parameter.numel())
                    gradient_chunk = gradient_flat[start:end]
                    if not torch.isfinite(gradient_chunk).all():
                        raise ValueError(
                            f"Objective {objective_id} produced a non-finite gradient"
                        )
                    current = gradient_chunk.to(
                        device="cpu", dtype=torch.float32
                    )
                    self.host_transfer_bytes += current.numel() * current.element_size()
                    # PyTorch minimizes pg_loss; negate it to store reward ascent.
                    tracker[objective_id, start:end].mul_(1.0 - self.beta).add_(
                        current, alpha=-self.beta
                    )
        if seen != set(self.parameter_shapes):
            raise RuntimeError("Objective gradient parameter set is incomplete")
        self._next_objective += 1

    def _local_gram(self, tensors: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        source = self.shards if tensors is None else tensors
        matrix = torch.zeros(
            (self.objective_count, self.objective_count), dtype=torch.float64
        )
        for tensor in source.values():
            for start in range(0, tensor.shape[1], self.gather_chunk_elements):
                flat = tensor[:, start:start + self.gather_chunk_elements].to(torch.float64)
                matrix += flat @ flat.T
        return matrix

    def direction_shard(self, name: str, coefficients: torch.Tensor | None = None) -> torch.Tensor:
        if self.last_step is None and coefficients is None:
            raise RuntimeError("Tracker direction is unavailable before finalization")
        weights = (
            self.last_step.solution.effective_coefficients
            if coefficients is None else coefficients
        ).to(dtype=torch.float32)
        return weights @ self.shards[name]

    def _bf16_diagnostics(
        self,
        reference: CAGradSolution,
        reference_gram: torch.Tensor,
        reduce_sum: ReduceSum,
    ) -> tuple[float, float, float]:
        local_gram = torch.zeros_like(reference_gram)
        for tensor in self.shards.values():
            for start in range(0, tensor.shape[1], self.gather_chunk_elements):
                quantized = tensor[:, start:start + self.gather_chunk_elements].to(
                    torch.bfloat16
                ).to(torch.float32).to(torch.float64)
                local_gram += quantized @ quantized.T
        quantized_gram = _validate_reduced(
            reduce_sum(local_gram), tuple(reference_gram.shape)
        )
        solution = cagrad_coefficients_from_gram(
            quantized_gram, radius=self.radius, resolution=self.resolution
        )
        gram_error = torch.linalg.vector_norm(quantized_gram - reference_gram).item()
        gram_error /= max(torch.linalg.vector_norm(reference_gram).item(), 1e-30)

        local = torch.zeros(3, dtype=torch.float64)
        for tensor in self.shards.values():
            for start in range(0, tensor.shape[1], self.gather_chunk_elements):
                chunk = tensor[:, start:start + self.gather_chunk_elements]
                reference_direction = reference.effective_coefficients.to(torch.float32) @ chunk
                quantized = chunk.to(torch.bfloat16).to(torch.float32)
                candidate_direction = solution.effective_coefficients.to(torch.float32) @ quantized
                local[0] += torch.dot(
                    reference_direction.double(), candidate_direction.double()
                )
                local[1] += reference_direction.double().square().sum()
                local[2] += candidate_direction.double().square().sum()
        products = _validate_reduced(reduce_sum(local), (3,))
        cosine = products[0].item() / max(
            math.sqrt(max(products[1].item(), 0.0) * max(products[2].item(), 0.0)),
            1e-30,
        )
        dual_l1 = torch.abs(solution.dual_weights - reference.dual_weights).sum().item()
        return gram_error, cosine, dual_l1

    def _streaming_gather_direction_error(
        self,
        coefficients: torch.Tensor,
        *,
        chunk_elements: int | None = None,
    ) -> tuple[float, int]:
        """Compare local directions with a true chunked gathered reference.

        The complete vector is logically concatenated rank by rank, but only a
        bounded chunk is resident on a GPU at once. This is equivalent to a
        monolithic gather for the global relative L2 error.
        """

        chunk_elements = self.gather_chunk_elements if chunk_elements is None else chunk_elements
        if chunk_elements < 1:
            raise ValueError("Gather chunk size must be positive")
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        backend = torch.distributed.get_backend() if distributed else "none"
        device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
        weights_cpu = coefficients.to(device="cpu", dtype=torch.float32)
        error = 0.0
        reference_norm = 0.0
        transfer_bytes = 0

        names = tuple(sorted(self.shards))
        if distributed:
            layout_digest = hashlib.sha256("\n".join(names).encode("utf-8")).digest()
            layout_token = torch.tensor(
                int.from_bytes(layout_digest[:8], "little", signed=False) % (2**63 - 1),
                dtype=torch.int64,
                device=device,
            )
            tokens = [torch.empty_like(layout_token) for _ in range(world_size)]
            torch.distributed.all_gather(tokens, layout_token)
            if len({int(item.item()) for item in tokens}) != 1:
                raise RuntimeError("FSDP tracker parameter names differ across ranks")

        for name in names:
            local_tracker = self.shards[name]
            local_length = local_tracker.shape[1]
            if distributed:
                length = torch.tensor(local_length, dtype=torch.int64, device=device)
                gathered_lengths = [torch.empty_like(length) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_lengths, length)
                lengths = [int(value.item()) for value in gathered_lengths]
            else:
                lengths = [local_length]
            maximum_length = max(lengths)
            for start in range(0, maximum_length, chunk_elements):
                width = min(chunk_elements, maximum_length - start)
                valid_local = max(0, min(width, local_length - start))
                tracker_input = torch.zeros(
                    (self.objective_count, width), dtype=torch.float32, device=device
                )
                direction_input = torch.zeros(width, dtype=torch.float32, device=device)
                if valid_local:
                    tracker_chunk = local_tracker[:, start:start + valid_local]
                    direction_chunk = weights_cpu @ tracker_chunk
                    tracker_input[:, :valid_local].copy_(tracker_chunk.to(device=device))
                    direction_input[:valid_local].copy_(direction_chunk.to(device=device))
                    if device.type == "cuda":
                        transfer_bytes += (tracker_chunk.numel() + direction_chunk.numel()) * 4

                if distributed:
                    gathered_tracker = [torch.empty_like(tracker_input) for _ in range(world_size)]
                    gathered_direction = [torch.empty_like(direction_input) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered_tracker, tracker_input)
                    torch.distributed.all_gather(gathered_direction, direction_input)
                else:
                    gathered_tracker = [tracker_input]
                    gathered_direction = [direction_input]

                if rank == 0:
                    weights_device = weights_cpu.to(device=device)
                    for source_rank, valid_length in enumerate(lengths):
                        valid = max(0, min(width, valid_length - start))
                        if not valid:
                            continue
                        reference = weights_device @ gathered_tracker[source_rank][:, :valid]
                        observed = gathered_direction[source_rank][:valid]
                        error += (observed.double() - reference.double()).square().sum().item()
                        reference_norm += reference.double().square().sum().item()

        totals = torch.tensor([error, reference_norm], dtype=torch.float64, device=device)
        if distributed:
            torch.distributed.broadcast(totals, src=0)
        relative_error = math.sqrt(max(totals[0].item(), 0.0)) / max(
            math.sqrt(max(totals[1].item(), 0.0)), 1e-30
        )
        return relative_error, transfer_bytes

    def _sharded_fp64_direction_error(
        self,
        coefficients: torch.Tensor,
        reduce_sum: ReduceSum,
    ) -> float:
        """Check the FP32 sharded direction against a chunked FP64 identity."""

        weights32 = coefficients.to(device="cpu", dtype=torch.float32)
        weights64 = coefficients.to(device="cpu", dtype=torch.float64)
        local = torch.zeros(2, dtype=torch.float64)
        for tensor in self.shards.values():
            for start in range(0, tensor.shape[1], self.gather_chunk_elements):
                chunk = tensor[:, start:start + self.gather_chunk_elements]
                direct = weights32 @ chunk
                reference = weights64 @ chunk.to(torch.float64)
                local[0] += (direct.double() - reference).square().sum()
                local[1] += reference.square().sum()
        totals = _validate_reduced(reduce_sum(local), (2,))
        return math.sqrt(max(totals[0].item(), 0.0)) / max(
            math.sqrt(max(totals[1].item(), 0.0)), 1e-30
        )

    def finalize(self, reduce_sum: ReduceSum = _identity_reduce) -> TrackerStep:
        if not self._active or self._next_objective != self.objective_count:
            raise RuntimeError("Tracker finalization requires every objective gradient")
        local_squared_norms = torch.zeros(self.objective_count, dtype=torch.float64)
        for tensor in self.shards.values():
            for start in range(0, tensor.shape[1], self.gather_chunk_elements):
                chunk = tensor[:, start:start + self.gather_chunk_elements].double()
                local_squared_norms += chunk.square().sum(dim=1)
        global_squared_norms = _validate_reduced(
            reduce_sum(local_squared_norms), (self.objective_count,)
        )
        if (global_squared_norms < -1e-12).any():
            raise ValueError("Reduced tracker norm is negative")
        norms = global_squared_norms.clamp_min(0.0).sqrt()
        scales = torch.minimum(
            torch.ones_like(norms),
            torch.full_like(norms, self.maximum_norm) / norms.clamp_min(1e-30),
        )
        for tensor in self.shards.values():
            tensor.mul_(scales.to(torch.float32).reshape(-1, 1))

        gram = _validate_reduced(
            reduce_sum(self._local_gram()),
            (self.objective_count, self.objective_count),
        )
        gram = 0.5 * (gram + gram.T)
        solution = cagrad_coefficients_from_gram(
            gram, radius=self.radius, resolution=self.resolution
        )

        next_step = self.step + 1
        diagnostics_audited = (
            self.diagnostic_steps is None or next_step in self.diagnostic_steps
        )
        if diagnostics_audited:
            gather_registered = (
                self.gather_reference_steps is None
                or next_step in self.gather_reference_steps
            )
            if gather_registered:
                reference_mode = "chunked_all_gather_full_direction"
                relative_error, gather_transfer_bytes = (
                    self._streaming_gather_direction_error(
                        solution.effective_coefficients
                    )
                )
            else:
                reference_mode = "sharded_fp64_global_l2"
                relative_error = self._sharded_fp64_direction_error(
                    solution.effective_coefficients, reduce_sum
                )
                gather_transfer_bytes = 0
            self.host_transfer_bytes += gather_transfer_bytes
            bf16_gram_error, bf16_cosine, bf16_dual_l1 = (
                self._bf16_diagnostics(solution, gram, reduce_sum)
            )
        else:
            reference_mode = "not_scheduled"
            relative_error = -1.0
            bf16_gram_error = -1.0
            bf16_cosine = -1.0
            bf16_dual_l1 = -1.0

        self.step += 1
        result = TrackerStep(
            step=self.step,
            gram=gram,
            solution=solution,
            projection=ProjectionDiagnostics(
                pre_projection_norms=norms,
                scales=scales,
                flags=norms > self.maximum_norm,
            ),
            diagnostics_audited=diagnostics_audited,
            reference_mode=reference_mode,
            direct_reference_relative_error=relative_error,
            bf16_gram_relative_error=bf16_gram_error,
            bf16_direction_cosine=bf16_cosine,
            bf16_dual_l1_error=bf16_dual_l1,
        )
        self.last_step = result
        self._active = False
        return result


def _optimizer_parameter_groups(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
) -> list[tuple[str, torch.nn.Parameter, dict]]:
    names = {id(parameter): name for name, parameter in named_parameters}
    output = []
    seen = set()
    for group in optimizer.param_groups:
        if group.get("maximize", False) or group.get("amsgrad", False):
            raise ValueError("The frozen preview requires ordinary minimizing AdamW")
        for parameter in group["params"]:
            if id(parameter) not in names or id(parameter) in seen:
                raise ValueError("Optimizer parameter layout is inconsistent with the actor")
            seen.add(id(parameter))
            output.append((names[id(parameter)], parameter, group))
    expected = {id(parameter) for _, parameter in named_parameters if parameter.requires_grad}
    if seen != expected:
        raise ValueError("Optimizer does not cover every trainable actor shard exactly once")
    return output


def preview_adamw_update(
    tracker: FullVectorTracker,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    *,
    gradient_clip: float,
    reduce_sum: ReduceSum = _identity_reduce,
    return_displacements: bool = False,
    chunk_elements: int = 1_048_576,
) -> tuple[dict, dict[str, torch.Tensor] | None]:
    """Preview the exact candidate AdamW displacement from live shared grads.

    ``parameter.grad`` contains the minimized shared regularizer loss only.
    Its sign is inverted before it is added to the reward-ascent direction.
    """

    if tracker.last_step is None or gradient_clip <= 0.0 or chunk_elements < 1:
        raise ValueError("A finalized tracker and positive clip are required")
    parameters = _optimizer_parameter_groups(named_parameters, optimizer)
    objective_count = tracker.objective_count
    coefficients = tracker.last_step.solution.effective_coefficients.to(torch.float32)
    local_summary = torch.zeros(4 + objective_count, dtype=torch.float64)
    for name, parameter, _ in parameters:
        tracker_shard = tracker.shards[name]
        if tracker_shard.shape[1] != parameter.numel():
            raise RuntimeError("Tracker and optimizer parameter shards differ in length")
        gradient = None if parameter.grad is None else parameter.grad.detach().reshape(-1)
        for start in range(0, parameter.numel(), chunk_elements):
            end = min(start + chunk_elements, parameter.numel())
            tracker_chunk = tracker_shard[:, start:end]
            direction = coefficients @ tracker_chunk
            if gradient is None:
                shared = torch.zeros_like(direction)
            else:
                gradient_chunk = gradient[start:end]
                if not torch.isfinite(gradient_chunk).all():
                    raise ValueError("Shared regularizer produced a non-finite gradient")
                shared = -gradient_chunk.to(device="cpu", dtype=torch.float32)
                tracker.host_transfer_bytes += shared.numel() * shared.element_size()
            raw = direction + shared
            local_summary[0] += raw.double().square().sum()
            local_summary[1] += direction.double().square().sum()
            local_summary[2] += shared.double().square().sum()
            local_summary[4:] += tracker_chunk.double() @ shared.double()
    summary = _validate_reduced(
        reduce_sum(local_summary), (4 + objective_count,)
    )
    raw_norm = math.sqrt(max(summary[0].item(), 0.0))
    clip_scale = min(1.0, gradient_clip / max(raw_norm, 1e-30))

    local_post = torch.zeros(5 + objective_count, dtype=torch.float64)
    displacements = {} if return_displacements else None
    step_values = []
    learning_rates = []
    for name, parameter, group in parameters:
        beta1, beta2 = tuple(float(value) for value in group["betas"])
        epsilon = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))
        learning_rate = float(group["lr"])
        state = optimizer.state.get(parameter, {})
        prior_step_value = state.get("step", 0)
        prior_step = int(
            prior_step_value.item() if torch.is_tensor(prior_step_value) else prior_step_value
        )
        step = prior_step + 1
        step_values.append(step)
        learning_rates.append(learning_rate)

        tracker_shard = tracker.shards[name]
        gradient = None if parameter.grad is None else parameter.grad.detach().reshape(-1)
        parameter_flat = parameter.detach().reshape(-1)
        previous_first_loss = state.get("exp_avg")
        previous_second = state.get("exp_avg_sq")
        if previous_first_loss is not None:
            if previous_second is None:
                raise RuntimeError("AdamW first and second moments are incomplete")
            previous_first_flat = previous_first_loss.detach().reshape(-1)
            previous_second_flat = previous_second.detach().reshape(-1)
        else:
            previous_first_flat = None
            previous_second_flat = None
        displacement_output = (
            torch.empty(parameter.numel(), dtype=torch.float32)
            if displacements is not None else None
        )

        for start in range(0, parameter.numel(), chunk_elements):
            end = min(start + chunk_elements, parameter.numel())
            tracker_chunk = tracker_shard[:, start:end]
            direction = coefficients @ tracker_chunk
            if gradient is None:
                shared = torch.zeros_like(direction)
            else:
                shared = -gradient[start:end].to(device="cpu", dtype=torch.float32)
                tracker.host_transfer_bytes += shared.numel() * shared.element_size()
            ascent = (direction + shared).mul(clip_scale).to(
                device=parameter.device, dtype=torch.float32
            )
            if previous_first_flat is None:
                previous_first_chunk = torch.zeros_like(ascent)
                previous_second_chunk = torch.zeros_like(ascent)
            else:
                previous_first_chunk = previous_first_flat[start:end].to(
                    device=parameter.device, dtype=torch.float32
                )
                previous_second_chunk = previous_second_flat[start:end].to(
                    device=parameter.device, dtype=torch.float32
                )
            second = beta2 * previous_second_chunk + (1.0 - beta2) * ascent.square()
            denominator = (second / (1.0 - beta2**step)).sqrt().add(epsilon)
            fresh = learning_rate * (
                (1.0 - beta1) * ascent / (1.0 - beta1**step) / denominator
            )
            history = learning_rate * (
                -beta1 * previous_first_chunk / (1.0 - beta1**step) / denominator
            )
            parameter_chunk = parameter_flat[start:end].to(torch.float32)
            decay = -learning_rate * weight_decay * parameter_chunk
            displacement = fresh + history + decay
            if not torch.isfinite(displacement).all():
                raise ValueError("AdamW preview produced a non-finite displacement")
            local_post[0] += displacement.double().square().sum().item()
            local_post[1] += fresh.double().square().sum().item()
            local_post[2] += history.double().square().sum().item()
            local_post[3] += decay.double().square().sum().item()
            local_post[4] += denominator.double().sum().item()
            displacement_cpu = displacement.to(device="cpu", dtype=torch.float32)
            tracker.host_transfer_bytes += (
                displacement_cpu.numel() * displacement_cpu.element_size()
            )
            local_post[5:] += tracker_chunk.double() @ displacement_cpu.double()
            if displacement_output is not None:
                displacement_output[start:end].copy_(displacement_cpu)
        if displacements is not None:
            displacements[name] = displacement_output
    post = _validate_reduced(reduce_sum(local_post), (5 + objective_count,))

    gram = tracker.last_step.gram
    coefficients = tracker.last_step.solution.effective_coefficients
    tracker_only_margins = gram @ coefficients
    shared_margins = summary[4:]
    raw_margins = tracker_only_margins + shared_margins
    record = {
        "raw_input_norm": raw_norm,
        "tracker_direction_norm": math.sqrt(max(summary[1].item(), 0.0)),
        "shared_direction_norm": math.sqrt(max(summary[2].item(), 0.0)),
        "clip_scale": clip_scale,
        "tracker_only_margins": tracker_only_margins.tolist(),
        "shared_margins": shared_margins.tolist(),
        "pre_adamw_margins": (raw_margins * clip_scale).tolist(),
        "post_adamw_margins": post[5:].tolist(),
        "adamw_displacement_norm": math.sqrt(max(post[0].item(), 0.0)),
        "adamw_fresh_norm": math.sqrt(max(post[1].item(), 0.0)),
        "adamw_history_norm": math.sqrt(max(post[2].item(), 0.0)),
        "adamw_weight_decay_norm": math.sqrt(max(post[3].item(), 0.0)),
        "adamw_denominator_sum": post[4].item(),
        "adamw_next_steps": sorted(set(step_values)),
        "learning_rates": sorted(set(learning_rates)),
        "adamw_preview_chunk_elements": chunk_elements,
    }
    return record, displacements


def execute_readonly_shadow_step(
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    tracker: FullVectorTracker,
    backward_objective: Callable[[int], None],
    backward_shared: Callable[[], None] | None,
    gradient_clip: float,
    reduce_sum: ReduceSum = _identity_reduce,
    return_displacements: bool = False,
    adamw_preview_chunk_elements: int = 1_048_576,
) -> tuple[dict, dict[str, torch.Tensor] | None]:
    """Execute one interleaved candidate shadow step and prove it is read-only."""

    named_parameters = list(module.named_parameters())
    before_rng = capture_rng_state()
    before_hashes = training_state_hashes(module, optimizer, scheduler)
    before_hashes["rng"] = rng_hash(before_rng)
    try:
        tracker.begin_step(named_parameters)
        for objective_id in range(tracker.objective_count):
            restore_rng_state(before_rng)
            optimizer.zero_grad(set_to_none=True)
            backward_objective(objective_id)
            tracker.absorb_loss_gradients(objective_id, named_parameters)
        tracker_step = tracker.finalize(reduce_sum=reduce_sum)

        restore_rng_state(before_rng)
        optimizer.zero_grad(set_to_none=True)
        if backward_shared is not None:
            backward_shared()
        preview, displacements = preview_adamw_update(
            tracker,
            named_parameters,
            optimizer,
            gradient_clip=gradient_clip,
            reduce_sum=reduce_sum,
            return_displacements=return_displacements,
            chunk_elements=adamw_preview_chunk_elements,
        )
    except Exception:
        tracker.poisoned = True
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(before_rng)

    after_rng = capture_rng_state()
    after_hashes = training_state_hashes(module, optimizer, scheduler)
    after_hashes["rng"] = rng_hash(after_rng)
    if before_hashes != after_hashes:
        tracker.poisoned = True
        changed = sorted(key for key in before_hashes if before_hashes[key] != after_hashes[key])
        raise RuntimeError("Read-only shadow modified training state: " + ", ".join(changed))

    record = {
        "tracker_step": tracker_step.step,
        "beta": tracker.beta,
        "horizon_actor_updates": tracker.horizon_actor_updates,
        "maximum_norm": tracker.maximum_norm,
        "radius": tracker.radius,
        "resolution": tracker.resolution,
        "gram": tracker_step.gram.tolist(),
        "dual_weights": tracker_step.solution.dual_weights.tolist(),
        "effective_coefficients": tracker_step.solution.effective_coefficients.tolist(),
        "pre_projection_norms": tracker_step.projection.pre_projection_norms.tolist(),
        "projection_scales": tracker_step.projection.scales.tolist(),
        "projection_flags": tracker_step.projection.flags.to(torch.int64).tolist(),
        "diagnostics_audited": tracker_step.diagnostics_audited,
        "reference_mode": tracker_step.reference_mode,
        "direct_reference_relative_error": tracker_step.direct_reference_relative_error,
        "bf16_gram_relative_error": tracker_step.bf16_gram_relative_error,
        "bf16_direction_cosine": tracker_step.bf16_direction_cosine,
        "bf16_dual_l1_error": tracker_step.bf16_dual_l1_error,
        "host_transfer_bytes_cumulative": tracker.host_transfer_bytes,
        "state_hashes_before": before_hashes,
        "state_hashes_after": after_hashes,
        "state_unchanged": True,
        **preview,
    }
    return record, displacements


def _load_active_minimization_gradients(
    tracker: FullVectorTracker,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    *,
    chunk_elements: int = 1_048_576,
) -> None:
    """Load ``shared_loss_gradient - reward_ascent_direction`` into ``.grad``."""

    if tracker.last_step is None or chunk_elements < 1:
        raise ValueError("A finalized tracker and positive chunk size are required")
    parameters = _optimizer_parameter_groups(named_parameters, optimizer)
    coefficients = tracker.last_step.solution.effective_coefficients.to(torch.float32)
    with torch.no_grad():
        for name, parameter, _ in parameters:
            tracker_shard = tracker.shards[name]
            if tracker_shard.shape[1] != parameter.numel():
                raise RuntimeError("Tracker and optimizer parameter shards differ in length")
            shared_gradient = parameter.grad
            if shared_gradient is not None and shared_gradient.is_sparse:
                raise ValueError("MoCo--CAGrad requires dense shared gradients")
            if shared_gradient is None:
                parameter.grad = torch.empty_like(
                    parameter, memory_format=torch.preserve_format
                )
            target = parameter.grad.detach().reshape(-1)
            shared_flat = (
                None if shared_gradient is None
                else shared_gradient.detach().reshape(-1)
            )
            for start in range(0, parameter.numel(), chunk_elements):
                end = min(start + chunk_elements, parameter.numel())
                direction = (
                    coefficients @ tracker_shard[:, start:end]
                ).to(device=target.device, dtype=target.dtype)
                if shared_flat is None:
                    target[start:end].copy_(direction, non_blocking=False).neg_()
                else:
                    target[start:end].sub_(direction)


def prepare_active_update(
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    tracker: FullVectorTracker,
    backward_objective: Callable[[int], None],
    backward_shared: Callable[[], None] | None,
    gradient_clip: float,
    reduce_sum: ReduceSum = _identity_reduce,
    preview_adamw: bool = False,
    return_displacements: bool = False,
    adamw_preview_chunk_elements: int = 1_048_576,
) -> tuple[dict, dict[str, torch.Tensor] | None]:
    """Prepare exactly one active MoCo--CAGrad optimizer update.

    Each objective sees the same stochastic masks.  On success, model,
    optimizer, and scheduler state are unchanged, while ``parameter.grad``
    contains the combined minimizing direction.  The caller must perform one
    ordinary clipped optimizer step and must not run a scalar PPO backward.
    """

    if return_displacements and not preview_adamw:
        raise ValueError("AdamW displacements require preview_adamw=True")
    named_parameters = list(module.named_parameters())
    before_rng = capture_rng_state()
    before_hashes = training_state_hashes(module, optimizer, scheduler)
    displacements = None
    try:
        tracker.begin_step(named_parameters)
        for objective_id in range(tracker.objective_count):
            restore_rng_state(before_rng)
            optimizer.zero_grad(set_to_none=True)
            backward_objective(objective_id)
            tracker.absorb_loss_gradients(objective_id, named_parameters)
        tracker_step = tracker.finalize(reduce_sum=reduce_sum)

        if backward_shared is not None:
            restore_rng_state(before_rng)
        optimizer.zero_grad(set_to_none=True)
        if backward_shared is not None:
            backward_shared()

        if preview_adamw:
            preview, displacements = preview_adamw_update(
                tracker,
                named_parameters,
                optimizer,
                gradient_clip=gradient_clip,
                reduce_sum=reduce_sum,
                return_displacements=return_displacements,
                chunk_elements=adamw_preview_chunk_elements,
            )
        else:
            coefficients = tracker_step.solution.effective_coefficients
            tracker_margins = tracker_step.gram @ coefficients
            preview = {
                "tracker_direction_norm": math.sqrt(max(
                    (coefficients @ tracker_step.gram @ coefficients).item(), 0.0
                )),
                "tracker_only_margins": tracker_margins.tolist(),
                "adamw_preview_performed": False,
            }

        _load_active_minimization_gradients(
            tracker,
            named_parameters,
            optimizer,
            chunk_elements=adamw_preview_chunk_elements,
        )
        prepared_hashes = training_state_hashes(module, optimizer, scheduler)
        if before_hashes != prepared_hashes:
            changed = sorted(
                key for key in before_hashes
                if before_hashes[key] != prepared_hashes[key]
            )
            raise RuntimeError(
                "Active preparation modified training state before optimizer: "
                + ", ".join(changed)
            )
    except Exception:
        tracker.poisoned = True
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(before_rng)
        raise

    record = {
        "update_mode": "active",
        "optimizer_step_applied": False,
        "gradient_loaded": True,
        "tracker_step": tracker_step.step,
        "beta": tracker.beta,
        "horizon_actor_updates": tracker.horizon_actor_updates,
        "maximum_norm": tracker.maximum_norm,
        "radius": tracker.radius,
        "resolution": tracker.resolution,
        "gram": tracker_step.gram.tolist(),
        "dual_weights": tracker_step.solution.dual_weights.tolist(),
        "effective_coefficients": (
            tracker_step.solution.effective_coefficients.tolist()
        ),
        "pre_projection_norms": (
            tracker_step.projection.pre_projection_norms.tolist()
        ),
        "projection_scales": tracker_step.projection.scales.tolist(),
        "projection_flags": (
            tracker_step.projection.flags.to(torch.int64).tolist()
        ),
        "diagnostics_audited": tracker_step.diagnostics_audited,
        "reference_mode": tracker_step.reference_mode,
        "direct_reference_relative_error": (
            tracker_step.direct_reference_relative_error
        ),
        "bf16_gram_relative_error": tracker_step.bf16_gram_relative_error,
        "bf16_direction_cosine": tracker_step.bf16_direction_cosine,
        "bf16_dual_l1_error": tracker_step.bf16_dual_l1_error,
        "host_transfer_bytes_cumulative": tracker.host_transfer_bytes,
        "state_hashes_before": before_hashes,
        "state_hashes_prepared": prepared_hashes,
        "state_unchanged_before_optimizer": True,
        "adamw_preview_performed": preview_adamw,
        **preview,
    }
    return record, displacements
