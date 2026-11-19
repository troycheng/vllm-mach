# SPDX-License-Identifier: Apache-2.0
"""Keep ordered GDN scratch bound across V2's final KV sizing."""
from dataclasses import fields
from functools import wraps


def bind_extensible_cache(worker, layers, view_key, scratch_by_view):
    cache = getattr(worker.model_runner, "extensible_kv_cache", None)
    if cache is None:
        return
    from vllm.v1.worker.gpu_worker import _min_across_ranks

    # The initial upper bound predates the persistent ordered scratch allocation.
    cache.warmup_committable_blocks = _min_across_ranks(cache.committable_blocks())
    original = worker.extend_kv_cache

    @wraps(original)
    def extend(num_blocks):
        before = [view_key(layer.kv_cache[1]) for layer in layers]
        result = original(num_blocks)
        after = [view_key(layer.kv_cache[1]) for layer in layers]
        rebound = {}
        for old, new in zip(before, after, strict=True):
            if (new[0] != old[0] or new[1][1:] != old[1][1:] or new[2:] != old[2:]
                    or not 2 <= new[1][0] <= old[1][0]):
                raise RuntimeError("Ordered GDN state pool address or layout changed after capture")
            if new not in rebound:
                scratch = scratch_by_view[old]
                pages = new[1][0]
                rebound[new] = type(scratch)(**{
                    field.name: getattr(scratch, field.name)[:pages]
                    for field in fields(scratch)
                })
        if len(rebound) != len(scratch_by_view):
            raise RuntimeError("Ordered GDN state sharing changed after capture")
        scratch_by_view.clear()
        scratch_by_view.update(rebound)
        return result

    worker.extend_kv_cache = extend
