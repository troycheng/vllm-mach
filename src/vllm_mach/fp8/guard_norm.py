# SPDX-License-Identifier: Apache-2.0
"""Source-bound R128 gated norm with direct strided gate input and output."""
import copy
import functools
import hashlib
import inspect as pyinspect
import json
from pathlib import Path
from types import FunctionType

from ..mxfp8.compile_choices import canonical_function_sha256
_BODY = 'e2a8f90402ecb470e6d0316f8cc77b832d675770f0187cb9308b33d31e38d65a'
_STATE = None


def clone_frontend(guard, kernel):
    """Keep allocation, contiguity, device context and reshape code unchanged.

    Private globals/launcher prevent a temporary process-global kernel swap.
    """
    def clone(fn, replacements):
        result = FunctionType(fn.__code__, {**fn.__globals__, **replacements},
                              fn.__name__, fn.__defaults__, fn.__closure__)
        result.__kwdefaults__ = fn.__kwdefaults__
        return result
    launcher = copy.copy(guard._LAYER_NORM_FWD_KERNEL)
    launcher.kernel = kernel
    forward = clone(guard.layer_norm_fwd, {'_LAYER_NORM_FWD_KERNEL': launcher})
    impl = clone(guard._layer_norm_fn_impl, {'layer_norm_fwd': forward})
    raw = clone(guard.rmsnorm_fn.__wrapped__, {'_layer_norm_fn_impl': impl})
    return guard.input_guard(raw)


def route(obj, args, kwargs, original):
    import torch
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as guard
    if getattr(obj, '_mach_fp8_ordered_reduction', False):
        obj._mach_fp8_python_runs += 1
    matched = getattr(obj, '_mach_fp8_guard_norm', None)
    if matched is None:
        fn = getattr(obj, 'fn', None)
        matched = fn is not None and canonical_function_sha256(fn.src) == _BODY
        if (not matched and fn is not None and 'rsqrt' in fn.__name__
                and isinstance(getattr(obj, 'size_hints', None), dict)
                and obj.size_hints.get('r0_') == 128):
            import os
            directory = os.environ.get('VLLM_MACH_FP8_RUN_DIR')
            body = canonical_function_sha256(fn.src)
            if directory:
                target = Path(directory) / 'unqualified-gdn-norm'
                target.mkdir(parents=True, exist_ok=True)
                (target / (body + '.py')).write_text(fn.src)
                from ..mxfp8.compile_choices import _config_fields
                configs = getattr(obj, 'configs', None)
                if configs is None:
                    configs = [x.config for x in getattr(obj, 'launchers', [])]
                record = dict(function_sha256=body, name=fn.__name__,
                              size_hints=obj.size_hints,
                              signature={str(k): str(v) for k, v in obj.triton_meta['signature'].items()},
                              actual_configs=[_config_fields(c) for c in configs])
                (target / (body + '.json')).write_text(json.dumps(record, indent=2) + '\n')
            raise RuntimeError('Unqualified block-FP8 R128 norm source: ' + body)
        obj._mach_fp8_guard_norm = matched
    if not matched:
        return original(obj, *args, **kwargs)
    flags = {k: obj.triton_meta.get(k, False) for k in
             ('enable_fp_fusion', 'disable_ftz', 'launch_pdl', 'native_matmul')}
    if flags != dict(enable_fp_fusion=True, disable_ftz=False, launch_pdl=False, native_matmul=False):
        raise RuntimeError('Block-FP8 gated norm compiler flags changed')
    fn = obj.fn
    names = [n for i, n in enumerate(fn.arg_names) if i not in fn.constexprs]
    if names != ['in_ptr0', 'in_ptr1', 'in_ptr2', 'out_ptr1', 'ks0', 'xnumel', 'r0_numel']:
        raise RuntimeError('Block-FP8 gated norm ABI changed')
    signature = pyinspect.Signature([pyinspect.Parameter(n, pyinspect.Parameter.POSITIONAL_OR_KEYWORD) for n in names])
    values = signature.bind(*args, **{k: v for k, v in kwargs.items() if k in names}).arguments
    t, m, n = (int(values[k]) for k in ('ks0', 'xnumel', 'r0_numel'))
    if t <= 0 or m != t * 32 or n != 128:
        raise RuntimeError('Block-FP8 gated norm geometry changed')
    rows = guard.calc_rows_per_block(m, values['in_ptr0'].device)
    if rows not in (1, 2, 4):
        _STATE['fallbacks'] += 1
        return original(obj, *args, **kwargs)
    def view(name, strides):
        x = values[name]
        shape = (t, 32, 128)
        end = x.storage_offset() + sum((d - 1) * s for d, s in zip(shape, strides))
        if x.dtype != torch.bfloat16 or end >= x.untyped_storage().nbytes() // x.element_size():
            raise RuntimeError('Block-FP8 gated norm tensor storage changed')
        return x.as_strided(shape, strides, x.storage_offset())
    x = view('in_ptr0', (4096, 128, 1))
    z = view('in_ptr2', (12288, 128, 1))
    out = view('out_ptr1', (4096, 128, 1))
    weight = values['in_ptr1']
    if weight.dtype != torch.bfloat16 or weight.numel() != 128 or weight.stride(0) != 1:
        raise RuntimeError('Block-FP8 gated norm weight changed')
    if x.is_cuda and kwargs['stream'] != torch.cuda.current_stream(x.device).cuda_stream:
        raise RuntimeError('Block-FP8 gated norm stream changed')
    from ._guard_direct import run
    run(x, weight, z, out)
    _STATE['hits'] += 1
    _STATE['rows'][str(rows)] = _STATE['rows'].get(str(rows), 0) + 1
    return None


def install():
    global _STATE
    if _STATE is not None:
        return False
    from .profile import NAME
    import os
    if os.environ.get('VLLM_MACH_PROFILE') != NAME:
        return False
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as guard
    from torch._inductor.runtime import triton_heuristics as heuristics
    gates = json.loads(Path(__file__).with_name('data').joinpath('legacy_math_sources.json').read_text())['guard_sources']
    key = 'vllm/third_party/flash_linear_attention/ops/layernorm_guard.py'
    if hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest() != gates[key]:
        raise RuntimeError('Block-FP8 gated norm frontend source changed')
    _STATE = dict(hits=0, fallbacks=0, rows={})
    for cls in [heuristics.CachingAutotuner, getattr(heuristics, 'StaticallyLaunchedAutotuner', None)]:
        if cls is None or 'run' not in cls.__dict__:
            continue
        previous = cls.run
        @functools.wraps(previous)
        def run(self, *args, _previous=previous, **kwargs):
            return route(self, args, kwargs, _previous)
        cls.run = run
    return True



def cache_receipt(kernel):
    """Inspect existing compiled JIT entries without creating a CUDA binder."""
    while not hasattr(kernel, 'device_caches'):
        kernel = kernel.fn
    rows = []
    for device, entry in list(kernel.device_caches.items()):
        for compiled in list(entry[0].values()):
            meta = compiled.metadata
            rows.append(dict(device=device, kernel_hash=compiled.hash,
                             constants={kernel.arg_names[k[0]]: v for k, v in compiled.src.constants.items()},
                             signature={str(k): str(v) for k, v in compiled.src.signature.items()},
                             target=dict(backend=meta.target.backend, arch=meta.target.arch),
                             metadata={k: getattr(meta, k) for k in ('num_warps', 'num_stages', 'shared', 'enable_fp_fusion', 'enable_reflect_ftz', 'launch_pdl') if hasattr(meta, k)},
                             jit_source_sha256=hashlib.sha256(kernel.src.encode()).hexdigest(),
                             cubin_sha256=hashlib.sha256(compiled.asm['cubin']).hexdigest()))
    return rows


def inspect():
    if _STATE is not None:
        from . import _guard_direct_kernel
    return dict(installed=_STATE is not None,
                actual_compiled=cache_receipt(_guard_direct_kernel.layer_norm_fwd_kernel) if _STATE is not None else [],
                hits=0 if _STATE is None else _STATE['hits'],
                fallbacks=0 if _STATE is None else _STATE['fallbacks'],
                rows_per_block_calls={} if _STATE is None else dict(_STATE['rows']),
                function_sha256=_BODY,
                kernel_source_sha256=hashlib.sha256(Path(__file__).with_name('_guard_direct_kernel.py').read_bytes()).hexdigest(),
                output_contract='Qualified R128 arithmetic with direct strided gate input and output store; original source/ABI/storage/stream guards preserved; counts exclude CUDA graph replays.')
