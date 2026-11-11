# SPDX-License-Identifier: Apache-2.0
"""Finite native MoE FP16 mixed GDN compatibility, preserving stock fallback."""
import copy
import hashlib
from types import FunctionType

_STATE = None
_WARMED = set()
_STATS = {'prepared': [], 'calls': {}, 'capture_calls': {}, 'fallback_calls': 0}
_ORIGINAL_JIT = 'f97378662341fe4f00dd2b51de78f49aebd7307f2126391db0dbd6d8129f5023'


def _eligible(layer, md, call, *, enabled, sm120):
    if (not enabled or not sm120 or not getattr(layer, '_mach_native_legacy', False)
            or layer.tp_size != 2 or layer.num_k_heads != 16 or layer.num_v_heads != 32
            or layer.head_k_dim != 128 or layer.head_v_dim != 128 or md is None):
        return False
    n = md.num_decodes
    if (not 1 <= n <= 32 or md.num_prefills <= 0 or md.num_decode_tokens != n
            or md.num_spec_decodes != 0 or md.spec_sequence_masks is not None):
        return False
    if (call.get('output') is not None or call.get('num_accepted_tokens') is not None
            or call.get('is_kda', False) or not call.get('use_qk_l2norm_in_kernel', False)
            or call.get('inplace_final_state', True) is not True):
        return False
    for key, shape in [('q', (1, n, 8, 128)), ('k', (1, n, 8, 128)),
                       ('v', (1, n, 16, 128)), ('a', (n, 16)), ('b', (n, 16)),
                       ('dt_bias', (16,))]:
        x = call.get(key)
        if x is None or tuple(x.shape) != shape or str(x.dtype) != 'torch.bfloat16' or not x.is_cuda:
            return False
    state = call.get('initial_state')
    if (state is not layer.kv_cache[1] or str(state.dtype) != 'torch.float16' or not state.is_cuda
            or tuple(state.shape[1:]) != (16, 128, 128)
            or state.stride() != (278528, 16384, 128, 1)):
        return False
    for key, length, exact in [('cu_seqlens', n + 1, True), ('ssm_state_indices', n, False)]:
        x = call.get(key)
        if (x is None or not x.is_cuda or str(x.dtype) != 'torch.int32' or len(x.shape) != 1
                or (x.shape[0] != length if exact else x.shape[0] < length) or x.stride() != (1,)):
            return False
    A = call.get('A_log')
    return A is not None and A.is_cuda and str(A.dtype) == 'torch.float32' and tuple(A.shape) == (16,)



def _clone_frontend(stock, kernel):
    name = 'fused_sigmoid_gating_delta_rule_update_kernel'
    private_kernel = copy.copy(stock.__globals__[name])
    private_kernel.fn = kernel
    private = FunctionType(stock.__code__, {**stock.__globals__, name: private_kernel},
                           stock.__name__, stock.__defaults__, stock.__closure__)
    private.__kwdefaults__ = stock.__kwdefaults__
    return private


def _frontend(stock):
    global _STATE
    from triton.runtime.jit import JITFunction
    if _STATE is not None:
        if stock is not _STATE['stock']:
            raise RuntimeError('Native mixed GDN frontend binding changed')
        return _STATE['private']
    name = 'fused_sigmoid_gating_delta_rule_update_kernel'
    original = stock.__globals__.get(name)
    if (not isinstance(getattr(original, 'fn', None), JITFunction)
            or stock.__name__ != 'fused_sigmoid_gating_delta_rule_update'
            or stock.__module__ != 'vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating'
            or hashlib.sha256(original.fn.src.encode()).hexdigest() != _ORIGINAL_JIT):
        raise RuntimeError('Native mixed GDN source/ABI changed')
    from .mixed_fp16_kernel import fused_sigmoid_gating_delta_rule_update_kernel as kernel
    if kernel.arg_names != original.fn.arg_names:
        raise RuntimeError('Native mixed GDN kernel arguments changed')
    private = _clone_frontend(stock, kernel)
    private_kernel = private.__globals__[name]
    _STATE = dict(stock=stock, private=private, original_kernel=original,
                  private_kernel=private_kernel, kernel=kernel)
    return private


def _warm(layer, frontend):
    import torch
    device = layer.A_log.device
    with torch.cuda.device(device):
        q = torch.zeros((1, 1, 8, 128), dtype=torch.bfloat16, device=device)
        v = torch.zeros((1, 1, 16, 128), dtype=torch.bfloat16, device=device)
        ab = torch.zeros((1, 16), dtype=torch.bfloat16, device=device)
        state = torch.empty_strided((2, 16, 128, 128), (278528, 16384, 128, 1),
                                   dtype=torch.float16, device=device)
        # Null slots compile the real varlen/pitch/dtype without touching live state.
        frontend(A_log=layer.A_log, a=ab, b=ab, dt_bias=layer.dt_bias,
                 q=q, k=q, v=v, initial_state=state, inplace_final_state=True,
                 cu_seqlens=torch.arange(2, dtype=torch.int32, device=device),
                 ssm_state_indices=torch.zeros(1, dtype=torch.int32, device=device),
                 use_qk_l2norm_in_kernel=True)


def prepare(layer):
    import torch
    from .legacy import enabled, _sm120
    if (not enabled() or not _sm120() or not getattr(layer, '_mach_native_legacy', False)
            or layer.tp_size != 2 or layer.num_k_heads != 16 or layer.num_v_heads != 32
            or layer.head_k_dim != 128 or layer.head_v_dim != 128
            or layer.get_state_dtype()[1] != torch.float16):
        return False
    if getattr(layer, '_mach_native_mixed_fp16', False):
        return False
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update as stock
    frontend = _frontend(stock)
    key = layer.A_log.device
    if key not in _WARMED:
        _warm(layer, frontend)
        _WARMED.add(key)
    layer._mach_native_mixed_fp16 = True
    _STATS['prepared'].append(layer.prefix)
    return True


def update(layer, metadata, stock, **values):
    from .legacy import enabled
    if (_STATE is None or stock is not _STATE['stock']
            or not getattr(layer, '_mach_native_mixed_fp16', False)
            or not _eligible(layer, metadata, values, enabled=enabled(), sm120=True)):
        _STATS['fallback_calls'] += 1
        return stock(**values)
    import torch
    prefix = layer.prefix
    _STATS['calls'][prefix] = _STATS['calls'].get(prefix, 0) + 1
    if torch.cuda.is_current_stream_capturing():
        _STATS['capture_calls'][prefix] = _STATS['capture_calls'].get(prefix, 0) + 1
    return _STATE['private'](**values)


def receipt():
    result = dict(installed=_STATE is not None, prepared=list(_STATS['prepared']),
                  calls=dict(_STATS['calls']), capture_calls=dict(_STATS['capture_calls']),
                  fallback_calls=_STATS['fallback_calls'],
                  scope='Host launches/captures, not graph replay counts. Native MoE TP2 FP16 mixed varlen only.')
    if _STATE is not None:
        from ...fp8.guard_norm import cache_receipt
        result.update(original_jit_sha256=_ORIGINAL_JIT,
                      candidate_jit_sha256=hashlib.sha256(_STATE['kernel'].src.encode()).hexdigest(),
                      original_frontend_bytecode_unchanged=_STATE['private'].__code__ is _STATE['stock'].__code__,
                      stock_kernel_binding_unchanged=_STATE['stock'].__globals__['fused_sigmoid_gating_delta_rule_update_kernel'] is _STATE['original_kernel'],
                      compiled=cache_receipt(_STATE['kernel']))
    return result
