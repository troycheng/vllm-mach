# SPDX-License-Identifier: Apache-2.0
"""Qualified native MoE FP16 reductions; unrelated paths stay upstream."""
import functools
import hashlib
import importlib
import json
import os
import threading
from pathlib import Path
from types import MethodType

_DATA = Path(__file__).parents[1] / 'profile'
_ENABLED = False
_INSTALLED = False
_CONTEXT = threading.local()
_WARMED = set()
_DECODE_ROWS = (1, 2, 4, 8, 16, 24, 32)
_PACKED_ROWS = (16, 24, 32)
_STATS = {'prepared': [], 'pins': {}, 'prep_calls': 0, 'norm_calls': {},
          'packed_calls': {}, 'packed_capture': {}, 'attention_calls': 0, 'attention_capture': {},
          'norm_choices': {}, 'norm_capture': {}, 'pin_choices': {}, 'attention_choices': []}


def enabled():
    return _ENABLED


def model_eligible(config):
    from ...runtime import attention
    if (os.environ.get('VLLM_QWEN3_5_FP16_SSM') != '1'
            or not attention._NATIVE_REGISTERED
            or getattr(config.cache_config, 'mamba_ssm_cache_dtype', None) != 'float16'):
        return False
    preserve_old_softmax_segments = attention.preserve_old_softmax_segments
    model = config.model_config
    text = model.hf_text_config
    return (not os.environ.get('VLLM_MACH_PROFILE')
            and preserve_old_softmax_segments(config)
            and text.model_type == 'qwen3_5_moe_text'
            and config.parallel_config.tensor_parallel_size == 2
            and config.parallel_config.pipeline_parallel_size == 1
            and config.parallel_config.decode_context_parallel_size == 1
            and not config.parallel_config.enable_dbo
            and config.speculative_config is None
            and config.cache_config.mamba_ssm_cache_dtype == 'float16'
            and all(getattr(text, key, None) == value for key, value in
                    {'hidden_size': 2048, 'num_hidden_layers': 40,
                     'linear_num_key_heads': 16, 'linear_num_value_heads': 32,
                     'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
                     'num_attention_heads': 16, 'num_key_value_heads': 2,
                     'head_dim': 256}.items()))


def _sm120():
    from vllm.platforms import current_platform
    return current_platform.is_cuda() and current_platform.is_device_capability(120)


def resolve_backend(config, requested):
    """Preserve explicit user choices, including an explicitly requested auto."""
    global _ENABLED
    if not model_eligible(config) or not _sm120():
        _ENABLED = False
        return requested
    additional = config.additional_config
    explicit = isinstance(additional, dict) and 'gdn_prefill_backend' in additional
    selected = requested if explicit else 'triton'
    _ENABLED = str(selected).strip().lower() == 'triton'
    if _ENABLED:
        install()
    return selected


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    from triton import Config
    from triton.runtime.autotuner import Autotuner
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
    gates = json.loads((_DATA / 'legacy_sources.json').read_text())
    for filename, digest in gates.items():
        module = importlib.import_module('vllm.third_party.flash_linear_attention.ops.' + filename[:-3])
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Native FLA source changed: ' + filename)
    for row in json.loads((_DATA / 'fla_choices.json').read_text())['kernels']:
        tuner = getattr(importlib.import_module(row['module']), row['function'])
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        original = tuner.run

        @functools.wraps(original)
        def run(self, *args, _row=row, _original=original, **kwargs):
            values = {**dict(zip(self.arg_names, args)), **kwargs}
            bound = {k: v for k, v in values.items() if k in self.arg_names}
            key = [bound[k] for k in self.keys if k in bound]
            key.extend(str(v.dtype) for v in bound.values() if hasattr(v, 'dtype'))
            if _ENABLED and key == _row['key']:
                choice = Config(**_row['selected'])
                if choice not in self.configs or (tuple(key) in self.cache and self.cache[tuple(key)] != choice):
                    raise RuntimeError('Native qualified FLA selection changed')
                self.cache[tuple(key)] = choice
            result = _original(*args, **kwargs)
            if _ENABLED and key == _row['key']:
                name = _row['function']
                _STATS['pins'][name] = _STATS['pins'].get(name, 0) + 1
                _STATS['pin_choices'][_row['function']] = dict(key=key, selected=_row['selected'])
                if hasattr(result, 'hash'):
                    _STATS['pin_choices'][_row['function']]['hash'] = result.hash
            return result
        tuner.run = MethodType(run, tuner)
    original_prep = qwen.fused_post_conv_prep

    @functools.wraps(original_prep)
    def prep(conv, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
             apply_l2norm=True, output_g_exp=False):
        match = (_ENABLED and num_k_heads == 8 and A_log.numel() == 16
                 and head_k_dim == head_v_dim == 128 and conv.is_cuda
                 and conv.dtype == torch.bfloat16 and conv.ndim == 2
                 and conv.shape[1] == 4096 and apply_l2norm is True
                 and output_g_exp is False)
        implementation = original_prep
        if match:
            from ..._gdn_prep_kernel import fused_post_conv_prep as implementation
        result = implementation(conv, a, b, A_log, dt_bias, num_k_heads,
                                head_k_dim, head_v_dim, apply_l2norm, output_g_exp)
        if match:
            _STATS['prep_calls'] += 1
        return result
    qwen.fused_post_conv_prep = prep
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as guard
    original_norm = guard.layer_norm_fwd_kernel.run

    def norm_run(*args, **kwargs):
        prefix = getattr(_CONTEXT, 'norm', None)
        if not _ENABLED or prefix is None:
            return original_norm(*args, **kwargs)
        expected = dict(BLOCK_N=128, HAS_BIAS=False, HAS_Z=True,
                        IS_RMS_NORM=True, NORM_BEFORE_GATE=True, num_warps=1)
        if (len(args) != 13 or args[11] != 128
                or kwargs.get('ROWS_PER_BLOCK') not in (1, 2, 4)
                or kwargs.get('ACTIVATION') not in ('silu', 'swish')
                or any(kwargs.get(k) != v for k, v in expected.items())):
            return original_norm(*args, **kwargs)
        from ...fp8._guard_kernel import layer_norm_fwd_kernel
        result = layer_norm_fwd_kernel.run(*args, **kwargs)
        key = prefix + ':row' + str(kwargs['ROWS_PER_BLOCK'])
        _STATS['norm_calls'][key] = _STATS['norm_calls'].get(key, 0) + 1
        if torch.cuda.is_current_stream_capturing():
            capture_key = key + ':m' + str(args[10])
            _STATS['norm_capture'][capture_key] = _STATS['norm_capture'].get(capture_key, 0) + 1
        _STATS['norm_choices'][key] = dict(rows=args[10], capturing=torch.cuda.is_current_stream_capturing(), hash=result.metadata.hash, num_warps=result.metadata.num_warps, num_stages=result.metadata.num_stages, ptx_sha256=hashlib.sha256(result.asm['ptx'].encode()).hexdigest())
        return result
    guard.layer_norm_fwd_kernel.run = norm_run
    _INSTALLED = True


def prepare(layer):
    """Bind only an already prepared native MoE instance before graph capture."""
    if not _ENABLED or getattr(layer, '_mach_native_legacy', False):
        return
    import torch
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as guard
    if (not getattr(layer, '_mach_gdn_prepared', False)
            or layer.tp_size != 2 or layer.in_proj_ba.weight.shape[1] != 2048
            or layer.num_k_heads != 16 or layer.num_v_heads != 32
            or layer.get_state_dtype()[1] != torch.float16
            or layer.gdn_prefill_backend != 'triton'
            or '-rms_norm_gated' in layer.vllm_config.compilation_config.custom_ops):
        return
    norm = layer.norm
    if (type(norm) is not RMSNormGated or norm.bias is not None
            or norm.weight.shape != (128,) or norm.weight.dtype != torch.bfloat16
            or norm.eps != layer.layer_norm_epsilon
            or norm.group_size is not None or norm.norm_before_gate is not True
            or norm.activation not in ('silu', 'swish')
            or getattr(norm._forward_method, '__func__', None) is not RMSNormGated.forward_native):
        return
    original = norm._forward_method

    def forward(x, z):
        if (layer.get_state_dtype()[1] != torch.float16 or x.shape != z.shape or x.shape[-1] != 128
                or x.dtype != z.dtype or x.dtype != torch.bfloat16
                or x.stride(-1) != 1 or z.stride(-1) != 1
                or guard.calc_rows_per_block(x.numel() // 128, x.device) not in (1, 2, 4)):
            return original(x, z)
        previous = getattr(_CONTEXT, 'norm', None)
        _CONTEXT.norm = layer.prefix
        try:
            return norm.forward_cuda(x, z)
        finally:
            _CONTEXT.norm = previous
    norm._forward_method = forward
    original_decode = layer._rms_norm_gated_cuda

    def decode_norm(x, z, out):
        if (layer.get_state_dtype()[1] != torch.float16 or x.ndim != 3 or x.shape[0] not in _DECODE_ROWS or x.shape[1:] != (16, 128) or z.shape != x.shape or out.shape != x.shape
                or x.dtype != z.dtype or x.dtype != out.dtype or x.dtype != torch.bfloat16):
            return original_decode(x, z, out)
        previous = getattr(_CONTEXT, 'norm', None)
        _CONTEXT.norm = layer.prefix
        try:
            return original_decode(x, z, out)
        finally:
            _CONTEXT.norm = previous
    layer._rms_norm_gated_cuda = decode_norm
    layer._mach_native_legacy = True
    _STATS['prepared'].append(layer.prefix)
    from . import packed
    # Empty indices compile the exact pitch/dtype without touching live slots.
    device = layer.in_proj_ba.weight.device
    if device.index in _WARMED:
        return
    mixed = torch.empty_strided((32, 4096), (6144, 1), dtype=torch.bfloat16, device=device)
    ab = torch.zeros((32, 16), dtype=torch.bfloat16, device=device)
    state = torch.empty_strided((2, 16, 128, 128), (278528, 16384, 128, 1), dtype=torch.float16, device=device)
    out = torch.empty((32, 1, 16, 128), dtype=torch.bfloat16, device=device)
    packed.decode(mixed, ab, ab, layer.A_log, layer.dt_bias, state, out,
                  torch.zeros(32, dtype=torch.int32, device=device))
    _WARMED.add(device.index)


def decode(layer, mixed, a, b, state, out, indices):
    import torch
    if (not _ENABLED or not getattr(layer, '_mach_native_legacy', False) or mixed.ndim != 2 or mixed.shape[0] not in _PACKED_ROWS or mixed.shape[1] != 4096
            or mixed.stride() != (6144, 1) or mixed.dtype != torch.bfloat16
            or state.dtype != torch.float16 or state.stride() != (278528, 16384, 128, 1)
            or a.shape != (mixed.shape[0], 16) or b.shape != a.shape
            or a.stride() != (16, 1) or b.stride() != (16, 1)
            or indices.shape != (mixed.shape[0],) or indices.dtype != torch.int32 or indices.stride() != (1,)):
        return False
    from . import packed
    packed.decode(mixed, a, b, layer.A_log, layer.dt_bias, state, out, indices)
    _STATS['packed_calls'][layer.prefix] = _STATS['packed_calls'].get(layer.prefix, 0) + 1
    if torch.cuda.is_current_stream_capturing():
        _STATS['packed_capture'][layer.prefix] = _STATS['packed_capture'].get(layer.prefix, 0) + 1
    return True


def prepare_attention(model):
    if not _ENABLED:
        return
    import torch
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    for layer in model.modules():
        impl = getattr(layer, 'impl', None)
        name = getattr(layer, 'layer_name', '')
        if (not isinstance(impl, TritonAttentionImpl) or '.layers.' not in name
                or not name.split('.layers.')[1].split('.')[0].isdigit()
                or int(name.split('.layers.')[1].split('.')[0]) not in range(3, 40, 4)
                or (impl.num_heads, impl.num_kv_heads, impl.head_size) != (8, 1, 256)
                or getattr(impl, '_mach_native_legacy', False)):
            continue
        original = impl.forward
        @functools.wraps(original)
        def forward(self, *args, _original=original, _name=name, _owner=layer, **kwargs):
            previous = getattr(_CONTEXT, 'attention', None)
            owner = kwargs.get('layer', args[0] if args else None)
            metadata = kwargs.get('attn_metadata', args[5] if len(args) > 5 else None)
            query = kwargs.get('query', args[1] if len(args) > 1 else None)
            admitted = (owner is _owner and owner.layer_name == _name
                        and metadata is not None and metadata.max_query_len == 1
                        and metadata.num_actual_tokens in _DECODE_ROWS
                        and metadata.seq_lens.shape == (metadata.num_actual_tokens,)
                        and query is not None and query.dtype == torch.bfloat16
                        and query.shape[0] == metadata.num_actual_tokens
                        and query.numel() == metadata.num_actual_tokens * 8 * 256)
            _CONTEXT.attention = _name if admitted else None
            try:
                return _original(*args, **kwargs)
            finally:
                _CONTEXT.attention = previous
        impl.forward = MethodType(forward, impl)
        impl._mach_native_legacy = True


def attention(stock, *, num_seqs, heads, kv_heads, segments, head_size, padded_head_size, bf16, use_fp8):
    if not (_ENABLED and getattr(_CONTEXT, 'attention', None) is not None
            and num_seqs in _DECODE_ROWS and heads == 8 and kv_heads == 1
            and segments == 16 and head_size == padded_head_size == 256 and bf16 and not use_fp8):
        return stock
    from ...mxfp8.two_b._attention_reduction_kernel import reduce_segments_explicit_old_lane as target
    class Launch:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                import torch
                result = target[grid](*args, **kwargs)
                prefix = _CONTEXT.attention
                _STATS['attention_calls'] += 1
                if torch.cuda.is_current_stream_capturing():
                    _STATS['attention_capture'][prefix] = _STATS['attention_capture'].get(prefix, 0) + 1
                row = dict(prefix=prefix, rows=num_seqs, grid=list(grid), capturing=torch.cuda.is_current_stream_capturing(), hash=result.metadata.hash, num_warps=result.metadata.num_warps, num_stages=result.metadata.num_stages, ptx_sha256=hashlib.sha256(result.asm['ptx'].encode()).hexdigest())
                if row not in _STATS['attention_choices']:
                    _STATS['attention_choices'].append(row)
                return result
            return run
    return Launch()


def receipt():
    from . import packed
    return dict(enabled=_ENABLED, installed=_INSTALLED, **_STATS,
                packed=packed.receipt(), counters='host launches/captures, not replays')
