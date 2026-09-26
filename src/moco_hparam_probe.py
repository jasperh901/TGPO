"""Explicit, narrow experimental forks; ordinary resume remains fail-closed."""
import json
import math
from pathlib import Path


def probe_spec(actor_config):
    value = actor_config.get('hparam_probe')
    return None if value is None else dict(value)


def validate_probe_source(spec, checkpoint, step):
    if not spec or Path(checkpoint).resolve() != Path(spec['source_checkpoint']).resolve():
        raise ValueError('Hyperparameter fork must use its pinned source checkpoint')
    if step != spec['source_step'] or step != 500:
        raise ValueError('Hyperparameter fork source step mismatch')
    if spec['source_horizon'] != 500 or spec['source_radius'] != 0.125:
        raise ValueError('Unexpected historical method parameters')


def validate_driver_fork(config, saved, checkpoint, contract_fn):
    from omegaconf import OmegaConf
    spec = probe_spec(config.actor_rollout_ref.actor)
    validate_probe_source(spec, checkpoint, saved['global_step'])
    if (saved['runtime_sha256'] != spec['source_runtime_sha256'] or
            saved['config_contract'] != spec['source_config_contract']):
        raise ValueError('Hyperparameter source provenance mismatch')
    # Reconstruct the original contract. Only these explicitly enumerated
    # differences are allowed; data, LR, KL, reward, rollout and GPU layout are not.
    original = OmegaConf.to_container(config, resolve=True)
    actor = original['actor_rollout_ref']['actor']
    actor.pop('hparam_probe')
    method = actor['moco_cagrad_shadow']
    if method['radius'] not in (0.125, 0.0625) or method['horizon_actor_updates'] not in (500, 100):
        raise ValueError('Candidate is outside the preregistered grid')
    method['radius'] = spec['source_radius']
    method['horizon_actor_updates'] = spec['source_horizon']
    if original['trainer']['total_training_steps'] != 532:
        raise ValueError('Unexpected quick-probe horizon')
    original['trainer']['total_training_steps'] = 500
    actor['optim']['total_training_steps'] = 500
    original['critic']['optim']['total_training_steps'] = 500
    if contract_fn(OmegaConf.create(original)) != saved['config_contract']:
        raise ValueError('Unapproved training change in hyperparameter fork')
    print('HPARAM_FORK_DRIVER=' + json.dumps({
        'source_step': 500, 'source': str(checkpoint),
        'source_runtime': saved['runtime_sha256'],
        'target_radius': config.actor_rollout_ref.actor.moco_cagrad_shadow.radius,
        'target_horizon': config.actor_rollout_ref.actor.moco_cagrad_shadow.horizon_actor_updates,
        'preserve_optimizer_rng_data_tracker': True,
        'is_unchanged_resume': False,
    }, sort_keys=True), flush=True)


def load_tracker_fork(tracker, state, named_parameters, spec, checkpoint, load_fn):
    validate_probe_source(spec, checkpoint, state['step'])
    source = state['config']
    if (source['horizon_actor_updates'] != spec['source_horizon'] or
            source['beta'] != 1 / math.sqrt(spec['source_horizon']) or
            source['radius'] != spec['source_radius']):
        raise ValueError('Unexpected source tracker parameters')
    target = (tracker.horizon_actor_updates, tracker.beta, tracker.radius)
    if target[0] not in (100, 500) or target[2] not in (0.0625, 0.125):
        raise ValueError('Tracker target is outside preregistered grid')
    try:
        tracker.horizon_actor_updates = source['horizon_actor_updates']
        tracker.beta = source['beta']
        tracker.radius = source['radius']
        # The existing strict loader still validates every remaining field,
        # shard ownership, shape, dtype and finiteness. No tracker reset.
        load_fn(tracker, state, named_parameters)
    finally:
        tracker.horizon_actor_updates, tracker.beta, tracker.radius = target
    print('HPARAM_FORK_TRACKER=' + json.dumps({
        'source_step': state['step'], 'source_beta': source['beta'],
        'target_beta': tracker.beta, 'source_radius': source['radius'],
        'target_radius': tracker.radius, 'retained_shards': len(tracker.shards),
    }, sort_keys=True), flush=True)
