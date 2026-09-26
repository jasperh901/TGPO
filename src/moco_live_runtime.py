"""Live diagnostics, applied only between serialized actor RPC calls."""
import json
import os
from pathlib import Path
import time


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def control(worker, action, directory):
    import torch
    actor = worker.actor
    tracker = actor.moco_cagrad_tracker
    if not actor.moco_cagrad_apply_update or tracker is None or tracker.poisoned or tracker._active:
        raise RuntimeError('Expected healthy, inactive exact MoCo-CAGrad tracker')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    info = {'rank': worker.rank, 'pid': os.getpid(), 'tracker_step': tracker.step,
            'torch_threads': torch.get_num_threads(), 'action': action,
            'tracker_bytes': sum(x.numel() * x.element_size() for x in tracker.shards.values())}
    if action == 'inspect':
        return info
    if action in ('validate_cache', 'validate_repeat'):
        return install_cache_validation(worker, root, info, repeat_only=action == 'validate_repeat')
    if action == 'checkpoint_hook':
        return install_checkpoint_hook(worker, info)
    if action != 'profile':
        raise ValueError(action)
    if getattr(actor, '_live_profile_pending', False):
        raise RuntimeError('A one-step profile is already pending')
    original = actor.moco_cagrad_execute
    actor._live_profile_pending = True

    def profiled(**kwargs):
        import moco_cagrad_shadow as core
        timings = {}

        def timed(name, function):
            def invoke(*args, **kw):
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    return function(*args, **kw)
                finally:
                    torch.cuda.synchronize()
                    timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
            return invoke

        attributes = ('absorb_loss_gradients', 'finalize', '_local_gram')
        old = {name: getattr(tracker, name) for name in attributes}
        globals_old = {name: getattr(core, name) for name in
                       ('training_state_hashes', '_load_active_minimization_gradients')}
        for name, fn in old.items():
            setattr(tracker, name, timed(name, fn))
        for name, fn in globals_old.items():
            setattr(core, name, timed(name, fn))
        objective = kwargs['backward_objective']
        kwargs['backward_objective'] = lambda index: timed(f'objective_{index}', objective)(index)
        if kwargs['backward_shared'] is not None:
            kwargs['backward_shared'] = timed('shared', kwargs['backward_shared'])
        start = time.perf_counter()
        try:
            return original(**kwargs)
        finally:
            for name, fn in old.items():
                setattr(tracker, name, fn)
            for name, fn in globals_old.items():
                setattr(core, name, fn)
            actor.moco_cagrad_execute = original
            actor._live_profile_pending = False
            payload = {**info, 'profiled_step': tracker.step, 'seconds': timings,
                       'total_seconds': time.perf_counter() - start}
            _write_json(root / f'profile_rank{worker.rank}.json', payload)
            print('moco_live_profile=' + json.dumps(payload), flush=True)

    actor.moco_cagrad_execute = profiled
    return info


def install_checkpoint_hook(worker, info):
    import types
    if getattr(worker, '_moco_checkpoint_hook', False):
        return {**info, 'already_installed': True}
    original = worker.save_checkpoint

    def save(this, local_path, hdfs_path=None):
        import torch
        from moco_checkpoint import save_worker_tracker
        # Write the method state before the original worker COMPLETE marker.
        # All ranks share this ordinary checkpoint boundary (no optimizer is
        # in progress) and the driver commits its state only after RPC returns.
        save_worker_tracker(this, Path(local_path) / 'training_state')
        torch.distributed.barrier()
        return original(local_path, hdfs_path)

    worker.save_checkpoint = types.MethodType(save, worker)
    worker._moco_checkpoint_hook = True
    return info


def install_cache_validation(worker, root, info, repeat_only=False):
    import functools
    import torch
    from moco_cagrad_shadow import capture_rng_state, restore_rng_state, rng_hash
    from moco_checkpoint import tracker_state_dict, load_tracker_state_dict
    from moco_fsdp_cache import prepare_cached_active_update
    actor = worker.actor
    tracker = actor.moco_cagrad_tracker
    if getattr(actor, '_live_profile_pending', False) or getattr(actor, '_cache_validation_pending', False):
        raise RuntimeError('Another diagnostic is pending')
    original = actor.moco_cagrad_execute
    candidate = original if repeat_only else functools.partial(
        prepare_cached_active_update, cache_budget_bytes=1610612736)
    actor._cache_validation_pending = True

    def validate(**kwargs):
        # The driver and rollout RNG never leave their live processes. Replay
        # only preparation on this identical already-generated actor batch.
        before = tracker_state_dict(tracker)
        before_rng = capture_rng_state()
        torch.cuda.synchronize()
        start = time.perf_counter()
        reference, reference_displacements = original(**kwargs)
        torch.cuda.synchronize()
        reference_seconds = time.perf_counter() - start
        reference_rng = capture_rng_state()
        reference_gradients = {name: p.grad.detach().cpu().clone() for name, p in
                               actor.actor_module.named_parameters() if p.requires_grad}
        expected_shards = tracker.shards
        expected_step = tracker.last_step
        expected_transfers = tracker.host_transfer_bytes
        load_tracker_state_dict(tracker, before, actor.actor_module.named_parameters())
        del before
        restore_rng_state(before_rng)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result, displacements = candidate(**kwargs)
        torch.cuda.synchronize()
        candidate_seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        exact_gradients = True
        max_abs_error = 0.0
        error_squared = reference_squared = 0.0
        for name, p in actor.actor_module.named_parameters():
            if not p.requires_grad:
                continue
            actual = p.grad.detach().cpu().reshape(-1)
            expected = reference_gradients[name].reshape(-1)
            exact_gradients &= torch.equal(actual, expected)
            for start in range(0, actual.numel(), 1048576):
                diff = actual[start:start+1048576].double() - expected[start:start+1048576].double()
                max_abs_error = max(max_abs_error, diff.abs().max().item())
                error_squared += diff.square().sum().item()
                reference_squared += expected[start:start+1048576].double().square().sum().item()
        exact_tracker = all(torch.equal(value, tracker.shards[name]) for name, value in expected_shards.items())
        exact_rng = rng_hash(capture_rng_state()) == rng_hash(reference_rng)
        exact_coefficients = reference['effective_coefficients'] == result['effective_coefficients']
        local_pass = exact_gradients and exact_tracker and exact_rng and exact_coefficients
        flag = torch.tensor(int(local_pass), device=torch.cuda.current_device())
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        passed = bool(flag.item())
        # Always restore the baseline branch after an A/B comparison, even if
        # tiny roundoff prevented exact equality. The caller updates only once.
        tracker.shards = expected_shards
        tracker.last_step = expected_step
        tracker.step = expected_step.step
        tracker.host_transfer_bytes = expected_transfers
        restore_rng_state(reference_rng)
        for name, p in actor.actor_module.named_parameters():
            if p.requires_grad:
                p.grad.copy_(reference_gradients[name].to(p.device))
        actor.moco_cagrad_execute = candidate if passed else original
        actor._cache_validation_pending = False
        payload = {**info, 'validated_step': tracker.step, 'status': 'PASS' if passed else 'REJECTED',
                   'exact_gradients': exact_gradients, 'exact_tracker': exact_tracker, 'exact_rng': exact_rng,
                   'exact_coefficients': exact_coefficients, 'max_abs_gradient_error': max_abs_error,
                   'relative_l2_gradient_error': (error_squared / max(reference_squared, 1e-300))**0.5,
                   'reference_seconds': reference_seconds, 'candidate_seconds': candidate_seconds,
                   'candidate_peak_allocated_gib': peak, 'cache': result.get('fixed_weight_cache')}
        filename = 'repeat_validation' if repeat_only else 'cache_validation'
        _write_json(root / f'{filename}_rank{worker.rank}.json', payload)
        print('moco_cache_validation=' + json.dumps(payload), flush=True)
        return reference, reference_displacements

    actor.moco_cagrad_execute = validate
    return info
