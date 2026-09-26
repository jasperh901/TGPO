"""Complete step-boundary snapshots for the pinned ToolRL runtime."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch


def release_cpu_allocator():
    """Return unused host arenas at I/O boundaries; live tensors are untouched."""
    import ctypes
    import gc
    gc.collect()
    libc = ctypes.CDLL(None)
    trim = getattr(libc, 'malloc_trim', None)
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)


def validate_resume_provenance(saved):
    # The launcher supplies one previous source digest only after a recorded,
    # structurally checked I/O-only migration. New snapshots use the current hash.
    current = os.environ.get('TRAIN_RUNTIME_SHA256')
    compatible = os.environ.get('TRAIN_COMPATIBLE_RUNTIME_SHA256')
    if not saved.get('runtime_sha256') or saved['runtime_sha256'] not in (current, compatible):
        raise ValueError('Resume provenance mismatch: runtime_sha256')
    if not saved.get('dataset_sha256') or saved['dataset_sha256'] != os.environ.get('TRAIN_DATASET_SHA256'):
        raise ValueError('Resume provenance mismatch: dataset_sha256')


def checkpoint_memory_event(worker, phase):
    import psutil
    from ray._private.utils import get_used_memory, get_system_memory
    print('CHECKPOINT_MEMORY ' + json.dumps({
        'rank': worker.rank, 'phase': phase,
        'rss_gib': psutil.Process().memory_info().rss / 2**30,
        'node_used_gib': get_used_memory() / 2**30,
        'node_total_gib': get_system_memory() / 2**30,
        'available_gib': psutil.virtual_memory().available / 2**30,
    }, sort_keys=True), flush=True)


def flush_checkpoint_file(path):
    """Durably flush only our file, then evict its clean, reproducible page cache."""
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def save_rank_states_serially(worker, local_path, complete_state):
    """Do not overlap eight tracker clones, serialization, and the full HF model.

    The worker RPC blocks updates and rollout. Capture and serialize one rank at
    a time, using the original state format and copying semantics. All ranks
    observe I/O errors before entering any subsequent FSDP collective.
    """
    import torch.distributed as dist
    release_cpu_allocator()
    dist.barrier()
    checkpoint_memory_event(worker, 'before_rank_states')
    for writer in range(worker.world_size):
        error = [None]
        if worker.rank == writer:
            try:
                Path(local_path).mkdir(parents=True, exist_ok=True)
                if complete_state:
                    runtime_state = capture_rollout(worker)
                    path = Path(local_path) / f'runtime_rank_{writer}.pt'
                    torch.save(runtime_state, path)
                    del runtime_state
                    flush_checkpoint_file(path)
                    release_cpu_allocator()
                path = Path(local_path) / f'optimizer_rank_{writer}.pt'
                torch.save({'optimizer': worker.actor_optimizer.state_dict(),
                            'scheduler': worker.actor_lr_scheduler.state_dict()}, path)
                flush_checkpoint_file(path)
                release_cpu_allocator()
                checkpoint_memory_event(worker, 'rank_states_written')
            except Exception as exc:
                error[0] = f'{type(exc).__name__}: {exc}'
        dist.broadcast_object_list(error, src=writer)
        if error[0] is not None:
            raise RuntimeError(f'Checkpoint rank {writer} failed: {error[0]}')
    release_cpu_allocator()
    dist.barrier()


def validate_complete_checkpoint_config(config, use_critic=False):
    """Accept only algorithms whose persistent state is fully serialized."""
    if not config.trainer.get('complete_checkpoint', False):
        return
    estimator = config.algorithm.adv_estimator
    moco = config.actor_rollout_ref.actor.get('moco_cagrad_shadow', {})
    active_moco = bool(moco.get('enabled', False))
    if not use_critic and estimator in ('gdpo', 'dvao') and not active_moco:
        return
    if (not use_critic and estimator == 'scppo_shadow' and active_moco
            and moco.get('apply_update', False)
            and int(moco.get('geometry_refresh_interval', 1)) == 1
            and not moco.get('adaptive_refresh_enabled', False)):
        return
    raise ValueError('Complete checkpoints support GDPO/DVAO and exact active MoCo-CAGrad only')


def capture_rng(cuda=True):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if cuda:
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'])


def runtime_versions():
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'vllm', 'transformers', 'ray')}
    versions['execution_env'] = {key: os.getenv(key) for key in (
        'FLASH_ATTENTION_DETERMINISTIC', 'CUBLAS_WORKSPACE_CONFIG',
        'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE')}
    return versions


def config_contract(config):
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(config, resolve=True)
    cfg['actor_rollout_ref']['actor'].pop('resume_from_checkpoint', None)
    for key in ('default_local_dir', 'experiment_name', 'logger', 'save_freq',
                'checkpoint_keep_latest', 'resume_step', 'complete_checkpoint',
                'stop_after_step', 'resume_audit_step', 'resume_audit_reference'):
        cfg['trainer'].pop(key, None)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def tensor_digest(tensor):
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), value.dtype)).encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def capture_rollout(worker):
    rollout = worker.rollout
    llm = rollout.inference_engine
    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError('Cannot checkpoint in-flight rollout requests')
    manager = worker.rollout_sharding_manager
    runner = engine.model_executor.worker.model_runner
    generators = getattr(runner, 'generators', None)
    if generators is None:
        generators = getattr(runner, '_generators', {})
    state = {
        'rng': capture_rng(),
        'gen_random_states': manager.gen_random_states,
        'torch_random_states': manager.torch_random_states,
        'request_counter': llm.request_counter.counter,
        'seq_counter': engine.seq_counter.counter,
        'generators': {key: gen.get_state() for key, gen in generators.items()},
        'num_gpu_blocks': engine.cache_config.num_gpu_blocks,
        'rank': worker.rank, 'world_size': worker.world_size,
        'versions': runtime_versions(),
    }
    tracker = getattr(worker.actor, 'moco_cagrad_tracker', None)
    if tracker is not None:
        if (worker.actor.moco_cagrad_geometry_refresh_interval != 1
                or worker.actor.moco_cagrad_adaptive_refresh_enabled):
            raise ValueError('Complete MoCo snapshots require exact per-update geometry')
        from moco_checkpoint import tracker_state_dict
        state['moco_tracker'] = tracker_state_dict(tracker)
    return state


def restore_rollout(worker, state):
    if (state['rank'], state['world_size']) != (worker.rank, worker.world_size):
        raise ValueError('Checkpoint rank/world size mismatch')
    if state['versions'] != runtime_versions():
        raise ValueError('Checkpoint library versions changed')
    tracker = getattr(worker.actor, 'moco_cagrad_tracker', None)
    if (tracker is not None) != ('moco_tracker' in state):
        raise ValueError('Checkpoint MoCo history missing or algorithm changed')
    if tracker is not None:
        from moco_checkpoint import load_tracker_state_dict
        load_tracker_state_dict(tracker, state['moco_tracker'],
                                worker.actor.actor_module.named_parameters())
        per_outer = worker.config.actor.moco_cagrad_shadow.actor_steps_per_outer
        if tracker.step != worker.actor_lr_scheduler.last_epoch * per_outer:
            raise ValueError('MoCo tracker and restored scheduler steps disagree')
    llm = worker.rollout.inference_engine
    engine = llm.llm_engine
    if engine.cache_config.num_gpu_blocks != state['num_gpu_blocks']:
        raise ValueError('Rollout KV block count changed')
    llm.request_counter.counter = state['request_counter']
    engine.seq_counter.counter = state['seq_counter']
    manager = worker.rollout_sharding_manager
    manager.gen_random_states = state['gen_random_states']
    manager.torch_random_states = state['torch_random_states']
    runner = engine.model_executor.worker.model_runner
    attr = 'generators' if hasattr(runner, 'generators') else '_generators'
    if state['generators'] and not hasattr(runner, attr):
        raise ValueError('Missing vLLM request generator state holder')
    if hasattr(runner, attr):
        setattr(runner, attr, {key: torch.Generator(device='cuda').set_state(value)
                               for key, value in state['generators'].items()})
    restore_rng(state['rng'])


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def commit_snapshot(root, pending, step, world_size):
    root, pending = Path(root), Path(pending)
    if pending.parent.resolve() != root.resolve() or not pending.name.startswith('.saving_'):
        raise ValueError('Pending checkpoint is outside its owned staging directory')
    required = ['trainer_state.pt', 'model.safetensors.index.json']
    if (pending / 'model.safetensors').is_file():
        required[1] = 'model.safetensors'
    for rank in range(world_size):
        required += [f'optimizer_rank_{rank}.pt', f'runtime_rank_{rank}.pt']
    for name in required:
        if not (pending / name).is_file() or not (pending / name).stat().st_size:
            raise ValueError(f'Incomplete checkpoint: {name}')
    index_path = pending / 'model.safetensors.index.json'
    if index_path.is_file():
        for name in set(json.loads(index_path.read_text())['weight_map'].values()):
            if Path(name).name != name or not (pending / name).is_file():
                raise ValueError(f'Missing model shard: {name}')
    sizes = {}
    for path in pending.iterdir():
        if path.is_file():
            sizes[path.name] = path.stat().st_size
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    atomic_json(pending / 'COMPLETE.json', {
        'schema': 1, 'global_step': step, 'world_size': world_size, 'files': sizes})
    descriptor = os.open(pending, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    final = root / f'global_step_{step}'
    if final.exists():
        raise FileExistsError(final)
    os.rename(pending, final)
    temp_link = root / '.latest.tmp'
    if temp_link.is_symlink():
        temp_link.unlink()
    temp_link.symlink_to(final.name, target_is_directory=True)
    os.replace(temp_link, root / 'latest')
    atomic_json(root / 'latest_checkpoint.json', {'global_step': step, 'path': 'latest'})
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    # Only remove this writer's older snapshots after the new pointer is durable.
    for path in root.iterdir():
        if path.is_dir() and not path.is_symlink() and path != final:
            if path.name.startswith('global_step_') and path.name[12:].isdigit():
                shutil.rmtree(path)
            elif path.name.startswith('.saving_'):
                shutil.rmtree(path)


def validate_snapshot(path):
    path = Path(path).resolve()
    manifest = json.loads((path / 'COMPLETE.json').read_text())
    for name, size in manifest['files'].items():
        if Path(name).name != name or (path / name).stat().st_size != size:
            raise ValueError(f'Checkpoint file size mismatch: {name}')
    return manifest
