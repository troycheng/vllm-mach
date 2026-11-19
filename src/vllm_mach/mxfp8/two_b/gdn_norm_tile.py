# SPDX-License-Identifier: Apache-2.0
"""Use the qualified 8-row tile for the vLLM 0.34 2B M16 gated norm."""
from collections import Counter
from copy import deepcopy
from functools import wraps

from vllm_mach.stack import BASE_PACKAGES
from ..compile_choices import _json_hash, descriptor

_KERNEL_PREFIX = 'triton_per_fused__to_copy_add_mean_mul_mxfp8_quantize_pow_rsqrt_silu_view_'
# Hash of the GDN norm function, signature, specialization, compiler flags,
# size hints and architecture as emitted by the pinned 0.34 production graph.
_SEMANTIC_KEY = '88cabbaa88ee6200911302a4b1571653256b7f8ad7dfe77242cbb2e4ea711174'
_ARGUMENTS = (16, 256, 128)  # ks0, xnumel, r0_numel
_INSTALLED = False
_COUNTS = Counter()


def _metadata(kernel):
    return {**kernel.triton_meta, 'device': kernel.device_props}


def _matches(kernel):
    if kernel.inductor_meta.get('grid_type') != 'Grid1D' or kernel.grid_mode != 'python':
        return False
    try:
        boundary = descriptor(kernel.fn.src, kernel.size_hints, _metadata(kernel))
    except (SyntaxError, TypeError, ValueError):
        return False
    return _json_hash(boundary) == _SEMANTIC_KEY


def _compile_variant(original, autotuner, config_type):
    # AsyncCompile normally keeps the JITFunction live in this profile. A
    # deserialized autotuner can reload it from its generated wrapper source.
    if original.fn.fn is None:
        original._ensure_kernel_loaded()
    config = config_type({'XBLOCK': 8}, num_warps=2, num_stages=1, num_ctas=1)
    meta = deepcopy(original.inductor_meta)
    for flag in ('coordinate_descent_tuning', 'dynamic_scale_rblock',
                 'dynamic_disable_pipelining', 'incremental_autotune'):
        meta[flag] = False
    variant = autotuner(original.fn, triton_meta=deepcopy(_metadata(original)),
                        configs=[config], save_cache_hook=None,
                        mutated_arg_names=list(original.mutated_arg_names),
                        reset_to_zero_arg_names=list(original.reset_to_zero_arg_names),
                        optimize_mem=original.optimize_mem, heuristic_type=original.heuristic_type,
                        size_hints=deepcopy(original.size_hints), inductor_meta=meta,
                        filename=original.filename)
    # An Inductor config lookup can override the constructor's input.
    variant.configs = [config]
    variant.precompile()
    if len(variant.launchers) != 1:
        raise RuntimeError('Expected one 2B M16 gated norm launcher')
    selected = variant.launchers[0].config
    if (selected.kwargs != {'XBLOCK': 8} or selected.num_warps != 2
            or selected.num_stages != 1 or selected.num_ctas != 1):
        raise RuntimeError('Unqualified 2B M16 gated norm launcher')
    return variant


class _M16Launcher:
    def __init__(self, original, variant, capturing):
        self._original = original
        self._variant = variant
        self._capturing = capturing

    def __getattr__(self, name):
        return getattr(self._original, name)

    def run(self, *args, **kwargs):
        if len(args) != 7 or args[4:] != _ARGUMENTS or kwargs.keys() != {'stream'}:
            return self._original.run(*args, **kwargs)
        result = self._variant.run(*args, **kwargs)
        _COUNTS['capture_calls' if self._capturing() else 'eager_calls'] += 1
        return result


def install():
    global _INSTALLED
    if _INSTALLED or BASE_PACKAGES['vllm'] != '0.34.0':
        return False

    import torch
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from triton import Config

    original_wait = AsyncCompile.wait
    capturing = torch.cuda.is_current_stream_capturing

    @wraps(original_wait)
    def wait(self, scope, *args, **kwargs):
        result = original_wait(self, scope, *args, **kwargs)
        for name, kernel in scope.items():
            if (name.startswith(_KERNEL_PREFIX) and name.removeprefix(_KERNEL_PREFIX).isdecimal()
                    and isinstance(kernel, CachingAutotuner) and _matches(kernel)):
                variant = _compile_variant(kernel, CachingAutotuner, Config)
                scope[name] = _M16Launcher(kernel, variant, capturing)
                _COUNTS['compiled_launchers'] += 1
        return result

    AsyncCompile.wait = wait
    _INSTALLED = True
    return True


def receipt():
    return {'installed': _INSTALLED, 'rows': 16, 'xblock': 8, 'num_warps': 2,
            'semantic_key': _SEMANTIC_KEY, 'compiled_launchers': _COUNTS['compiled_launchers'],
            'capture_calls': _COUNTS['capture_calls'], 'eager_calls': _COUNTS['eager_calls']}
