"""Bounded full-parameter reuse inside a fixed-weight MoCo preparation.

This never delays gradient reduction or changes its dtype/order. Full BF16
weights are retained only between backwards at identical parameter values,
and released before the caller may update or checkpoint any parameter.
"""
from contextlib import AbstractContextManager
import types

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class FixedWeightFSDPCache(AbstractContextManager):
    def __init__(self, module, budget_bytes):
        if budget_bytes < 0:
            raise ValueError('Cache budget must be nonnegative')
        self.module = module
        self.budget_bytes = budget_bytes
        self.selected = []
        self.bytes = 0
        self.suppressed_frees = 0
        self.originals = []
        self.active = False

    def __enter__(self):
        if self.active:
            raise RuntimeError('Cache context is already active')
        if torch.__version__.split('+')[0] != '2.4.0':
            raise RuntimeError('Fixed-weight cache requires revalidation outside PyTorch 2.4.0')
        self.selected = []
        self.originals = []
        self.bytes = 0
        self.suppressed_frees = 0
        candidates = []
        for name, child in self.module.named_modules():
            if not isinstance(child, FSDP) or child._handle is None:
                continue
            handle = child._handle
            if handle._use_orig_params or handle._offload_params:
                raise ValueError('Cache requires sharded flat parameters without CPU offload')
            if getattr(handle, '_moco_cache_owner', None) is not None:
                raise RuntimeError('Nested fixed-weight caches are forbidden')
            full = handle.flat_param._full_param_padded
            size = full.numel() * full.element_size()
            candidates.append((size, name, handle))
        for size, name, handle in sorted(candidates, key=lambda item: (item[0], item[1])):
            if size + self.bytes <= self.budget_bytes:
                self.selected.append((name, handle))
                self.bytes += size
        # This selection depends only on the identical model layout and budget,
        # never on rank-local available memory or response length.
        self.active = True
        for name, handle in self.selected:
            original = handle.reshard
            self.originals.append((handle, original))
            handle._moco_cache_owner = self

            def retain(this, free_unsharded_flat_param, original=original):
                if free_unsharded_flat_param:
                    self.suppressed_frees += 1
                return original(False)

            handle.reshard = types.MethodType(retain, handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available() and next(self.module.parameters()).is_cuda:
            torch.cuda.synchronize()
        for handle, original in self.originals:
            handle.reshard = original
            handle._moco_cache_owner = None
            original(True)
            handle.post_reshard()
            handle._prefetched = False
        self.active = False
        return False

    def summary(self):
        return {'cache_bytes': self.bytes, 'cache_handles': [name for name, _ in self.selected],
                'suppressed_frees': self.suppressed_frees, 'released': not self.active}


def prepare_cached_active_update(*, cache_budget_bytes, **kwargs):
    from moco_cagrad_shadow import prepare_active_update
    cache = FixedWeightFSDPCache(kwargs['module'], cache_budget_bytes)
    with cache:
        record, displacements = prepare_active_update(**kwargs)
    record['fixed_weight_cache'] = cache.summary()
    return record, displacements
