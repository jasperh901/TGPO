"""Versioned snapshots for the persistent, rank-local exact MoCo state."""
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch


CONFIG_KEYS = ('objective_count', 'horizon_actor_updates', 'beta', 'maximum_norm',
               'radius', 'resolution', 'gather_reference_steps', 'diagnostic_steps',
               'gather_chunk_elements')


def tracker_state_dict(tracker):
    if tracker.poisoned or tracker._active:
        raise RuntimeError('Cannot checkpoint a failed or unfinished MoCo update')
    return {
        'version': 1,
        'config': {key: getattr(tracker, key) for key in CONFIG_KEYS},
        'step': tracker.step,
        'parameter_shapes': dict(tracker.parameter_shapes),
        'shards': {key: value.detach().clone() for key, value in tracker.shards.items()},
        'last_step': copy.deepcopy(tracker.last_step),
        'host_transfer_bytes': tracker.host_transfer_bytes,
    }


def load_tracker_state_dict(tracker, state, named_parameters):
    if tracker.poisoned or tracker._active:
        raise RuntimeError('Cannot restore an active or failed MoCo tracker')
    if state.get('version') != 1:
        raise ValueError('Unsupported MoCo checkpoint version')
    config = {key: getattr(tracker, key) for key in CONFIG_KEYS}
    if state['config'] != config:
        raise ValueError('MoCo algorithm or execution configuration differs')
    observed = {name: tuple(parameter.shape) for name, parameter in named_parameters
                if parameter.requires_grad}
    if state['parameter_shapes'] != observed:
        raise ValueError('MoCo FSDP parameter ownership differs')
    if state['step'] < 1 or state['last_step'].step != state['step']:
        raise ValueError('Invalid MoCo step counters')
    if set(state['shards']) != set(observed):
        raise ValueError('Incomplete MoCo parameter shards')
    restored = {}
    for name, tensor in state['shards'].items():
        count = 1
        for dimension in observed[name]:
            count *= dimension
        if tensor.shape != (tracker.objective_count, count) or tensor.dtype != torch.float32:
            raise ValueError(f'Invalid MoCo tensor layout: {name}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite MoCo checkpoint: {name}')
        restored[name] = tensor.detach().to('cpu').clone()
    tracker.shards = restored
    tracker.parameter_shapes = dict(observed)
    tracker.step = state['step']
    tracker.last_step = copy.deepcopy(state['last_step'])
    tracker.host_transfer_bytes = state['host_transfer_bytes']
    tracker._next_objective = tracker.objective_count


def save_worker_tracker(worker, state_dir):
    tracker = getattr(worker.actor, 'moco_cagrad_tracker', None)
    if tracker is None:
        return
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {'rank': worker.rank, 'world_size': torch.distributed.get_world_size(),
               'tracker': tracker_state_dict(tracker), 'python': random.getstate(),
               'numpy': np.random.get_state()}
    path = state_dir / f'moco_rank{worker.rank}.pt'
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_worker_tracker(worker, state_dir):
    tracker = getattr(worker.actor, 'moco_cagrad_tracker', None)
    if tracker is None:
        return
    path = Path(state_dir) / f'moco_rank{worker.rank}.pt'
    if not path.is_file():
        raise RuntimeError(f'Missing MoCo history; cannot resume exact method: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['rank'] != worker.rank or state['world_size'] != torch.distributed.get_world_size():
        raise ValueError('MoCo checkpoint rank layout differs')
    from moco_hparam_probe import probe_spec, load_tracker_fork
    spec = probe_spec(worker.config.actor) if hasattr(worker, 'config') else None
    if spec and Path(state_dir).resolve() == Path(spec['source_checkpoint']).resolve():
        load_tracker_fork(tracker, state['tracker'], worker.actor.actor_module.named_parameters(),
                          spec, state_dir, load_tracker_state_dict)
    else:
        load_tracker_state_dict(tracker, state['tracker'], worker.actor.actor_module.named_parameters())
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
