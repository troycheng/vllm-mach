# SPDX-License-Identifier: Apache-2.0
"""TP2 Qwen3.8 BA scheduling: M32, plus opt-in FP16-state M16/M24."""
import os
import torch

from .mxfp6_hybrid import HybridRoute, state_for_rows


def enabled():
    return os.getenv('VLLM_MACH_BA_OVERLAP', '0') == '1'


def merged_m32_route(layer):
    state = state_for_rows(layer, 32)
    return (state is not None and state.route is HybridRoute.PREFILL_AND_M32
            and state.merged_weight is not None)


_AUX = None
_WARMED = False
_EXTRA_WARMED_ROWS = set()
FP16_SSM_ENV = 'VLLM_MACH_FP16_SSM_STATE'
EXTRA_ROWS_ENV = 'VLLM_MACH_BA_OVERLAP_M16_M24'


def narrow_fp16_state_allowed(module):
    return (
        os.getenv(FP16_SSM_ENV) == '1'
        and getattr(module, '_mach_fp16_ssm_decode_enabled', False) is True
        and module.kv_cache[0].dtype == torch.bfloat16
        and module.kv_cache[1].dtype == torch.float16
        and module.tp_size == 2 and module.hidden_size == 5120
        and module.num_k_heads == 16 and module.num_v_heads == 48
        and module.head_k_dim == 128 and module.head_v_dim == 128
        and module.enable_fused_gdn_decode and module.enable_packed_recurrent_decode
    )


def extra_row_allowed(module, hidden):
    rows = hidden.shape[0] if hidden.ndim == 2 else -1
    return (
        os.getenv(EXTRA_ROWS_ENV) == '1'
        and rows in (16, 24) and hidden.shape == (rows, 5120)
        and hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
        and narrow_fp16_state_allowed(module)
        and not module.gqa_interleaved_layout
        and module.norm.weight.dtype in (torch.bfloat16, torch.float32)
        # The checkpoint route retains EXL3 at M16/M24 and merges QKV at M32.
        and merged_m32_route(module.in_proj_qkvz)
        and state_for_rows(module.in_proj_qkvz, rows) is None
        and module._fi_fused_decode_step is not None
        and module._fi_fused_decode_supported is not None
    )


def prepare_stream():
    global _AUX
    if not enabled():
        return
    if _AUX is None:
        assert not torch.cuda.is_current_stream_capturing()
        _AUX = torch.cuda.Stream(device=torch.cuda.current_device())


def before_qkv(module, hidden):
    global _WARMED
    if not enabled():
        return None
    assert getattr(module, '_mach_ba_pending', None) is None
    rows = hidden.shape[0] if hidden.ndim == 2 else -1
    # Warm the existing cuBLAS BA path once on the auxiliary stream before
    # graph construction. This is outside every scored request lifecycle.
    if not _WARMED and not torch.cuda.is_current_stream_capturing() and hidden.shape[0] >= 32:
        main = torch.cuda.current_stream()
        _AUX.wait_stream(main)
        with torch.cuda.stream(_AUX):
            warm, _ = module.in_proj_ba(hidden[:32])
        main.wait_stream(_AUX)
        warm.record_stream(main)
        _WARMED = True
    extra_row = extra_row_allowed(module, hidden)
    if (extra_row and rows not in _EXTRA_WARMED_ROWS
            and not torch.cuda.is_current_stream_capturing()):
        assert _AUX is not None
        main = torch.cuda.current_stream()
        _AUX.wait_stream(main)
        with torch.cuda.stream(_AUX):
            warm, _ = module.in_proj_ba(hidden)
        main.wait_stream(_AUX)
        warm.record_stream(main)
        _EXTRA_WARMED_ROWS.add(rows)
    existing_m32 = hidden.shape == (32, 5120) and merged_m32_route(module.in_proj_qkvz)
    if (not (existing_m32 or extra_row) or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous()
            or module.tp_size != 2 or module.gqa_interleaved_layout
            or not module.enable_fused_gdn_decode or not module.enable_packed_recurrent_decode
            or module.norm.weight.dtype not in (torch.bfloat16, torch.float32)
            or module.head_k_dim != 128 or module.head_v_dim != 128
            or module.num_v_heads != 48 or module.num_k_heads != 16
            or module._fi_fused_decode_step is None or module._fi_fused_decode_supported is None):
        return None
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    raw = get_forward_context().attn_metadata
    if not isinstance(raw, dict) or module.prefix not in raw:
        return None
    meta = raw[module.prefix]
    if (meta.spec_sequence_masks is not None or meta.num_prefills != 0 or meta.num_decodes <= 0
            or meta.num_actual_tokens != rows or module.kv_cache[0].dtype != torch.bfloat16
            or not (module.kv_cache[1].dtype == torch.float32
                    or narrow_fp16_state_allowed(module))):
        return None
    supported = module._fi_fused_decode_supported(
        rows, hidden_size=module.hidden_size, n_ba=module.in_proj_ba.output_size_per_partition,
        qkv_dim=module.conv_dim // module.tp_size,
        num_qk_heads=module.num_k_heads // module.tp_size,
        num_v_heads=module.num_v_heads // module.tp_size, head_dim=module.head_k_dim,
        conv_width=module.conv_kernel_size, conv_state_len=module.conv_kernel_size - 1,
        device=hidden.device, conv_state_layout='DS' if is_conv_state_dim_first() else 'SD')
    # M16 is registered by FI, but _forward_core_fi rejects FP16 state.
    # Only the narrow extra-row path may bypass that geometry-only answer.
    if supported and not extra_row:
        return None
    assert _AUX is not None
    assert _WARMED if rows == 32 else rows in _EXTRA_WARMED_ROWS
    main = torch.cuda.current_stream()
    reference = None
    if os.getenv('VLLM_MACH_BA_OVERLAP_VERIFY', '0') == '1':
        qkv, _ = module.in_proj_qkvz(hidden)
        ba, _ = module.in_proj_ba(hidden)
        b, a = module.split_ba(ba)
        reference = (qkv, ba, b.contiguous(), a.contiguous())
    _AUX.wait_stream(main)
    return {'main': main, 'aux': _AUX, 'reference': reference, 'rows': rows}


def after_qkv(module, hidden, pending, qkv=None):
    if pending is None:
        return
    with torch.cuda.stream(pending['aux']):
        ba, _ = module.in_proj_ba(hidden)
        b, a = module.split_ba(ba)
        b, a = b.contiguous(), a.contiguous()
    pending.update(ba=ba, b=b, a=a)
    if pending.get('reference') is not None:
        if qkv is None:
            raise RuntimeError('BA verification requires the QKV result')
        torch._assert_async(torch.all(qkv.view(torch.int16) == pending['reference'][0].view(torch.int16)),
                            'BA overlap QKV differs from serial reference')
    module._mach_ba_pending = pending
    # Keep every captured producer buffer alive for the graph's lifetime.
    # Eager calls are protected by normal cross-stream allocator tracking.
    if torch.cuda.is_current_stream_capturing():
        if not hasattr(module, '_mach_ba_graph_buffers'):
            module._mach_ba_graph_buffers = []
        module._mach_ba_graph_buffers.append((ba, b, a, pending.get('reference')))
    else:
        for t in (ba, b, a):
            t.record_stream(pending['main'])


def select_ba(module, hidden):
    pending = getattr(module, '_mach_ba_pending', None)
    if pending is None:
        return module.in_proj_ba(hidden)
    return pending['ba'], None


def split_ba(module, ba):
    pending = getattr(module, '_mach_ba_pending', None)
    if pending is None:
        return module.split_ba(ba)
    assert ba is pending['ba']
    return pending['b'], pending['a']


def before_recurrent(module):
    pending = getattr(module, '_mach_ba_pending', None)
    if pending is None:
        return
    main = torch.cuda.current_stream()
    assert main.cuda_stream == pending['main'].cuda_stream
    main.wait_stream(pending['aux'])
    if pending.get('reference') is not None:
        for actual, expected in zip((pending['ba'], pending['b'], pending['a']), pending['reference'][1:]):
            torch._assert_async(torch.all(actual.view(torch.int16) == expected.view(torch.int16)),
                                'BA overlap branch differs from serial reference')
        if torch.cuda.is_current_stream_capturing() and not getattr(module, '_mach_ba_verified_graph', False):
            print('MACH_BA_VERIFY_GRAPH rank=%d prefix=%s' % (module.tp_rank, module.prefix), flush=True)
            module._mach_ba_verified_graph = True
    module._mach_ba_pending = None
    rows = pending.get('rows', 32)
    recorded = getattr(module, '_mach_ba_recorded_rows', set())
    if torch.cuda.is_current_stream_capturing() and rows not in recorded:
        print('MACH_BA_OVERLAP_CAPTURE rank=%d prefix=%s M=%d join=before_packed_recurrent' %
              (module.tp_rank, module.prefix, rows), flush=True)
        module._mach_ba_recorded_rows = recorded | {rows}
