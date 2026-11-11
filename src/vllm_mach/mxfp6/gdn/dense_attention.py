# SPDX-License-Identifier: Apache-2.0
"""Qualified Dense FP32 M32 NT16 merge; other native paths keep their choice."""
import functools
import hashlib
import importlib
import threading
from pathlib import Path
from types import MethodType

from . import dense_prep

_ENABLED = False
_VERIFIED = False
_CONTEXT = threading.local()
_WARMED = set()
_STATS = {'prepared': [], 'calls': {}, 'capture_calls': {}, 'kernels': {}, 'warmups': {}}


def configure(config):
    global _ENABLED
    text = config.model_config.hf_text_config
    _ENABLED = (dense_prep.model_eligible(config) and dense_prep._sm120()
                and config.cache_config.mamba_ssm_cache_dtype == 'float32'
                and all(getattr(text, k, None) == v for k, v in
                        {'num_attention_heads': 24, 'num_key_value_heads': 4,
                         'head_dim': 256}.items()))


def _target():
    global _VERIFIED
    from ...mxfp8.two_b._attention_reduction_kernel import reduce_segments_explicit_old_lane

    if not _VERIFIED:
        module = importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
        helper = importlib.import_module('vllm.v1.attention.ops.triton_attention_helpers')
        if (hashlib.sha256(module.reduce_segments.src.encode()).hexdigest()
                != '8d8e649d843c7a4d8b721123b2ee5a8348760e9cb138ede45cf2b6f7f0ab193e'
                or hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()
                != 'ef697fa4fa19c2543d9001c1db6cf6e8c5da22dd36ec5c1bc29f6a9ac8a59da9'
                or hashlib.sha256(reduce_segments_explicit_old_lane.src.encode()).hexdigest()
                != '50e2e8953a0393d067ebabf7fdfee2072ade0826b43d590357d5dea24295258f'):
            raise RuntimeError('Dense attention merge source changed')
        _VERIFIED = True
    return reduce_segments_explicit_old_lane


def _remember(kernel):
    if kernel.hash not in _STATS['kernels']:
        _STATS['kernels'][kernel.hash] = dict(
            num_warps=kernel.metadata.num_warps, num_stages=kernel.metadata.num_stages,
            ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest())


def _warmup(device):
    if device in _WARMED:
        return
    import torch

    target = _target()
    out = torch.empty((32, 12, 256), dtype=torch.bfloat16, device=device)
    staging = torch.zeros((32, 12, 16, 256), dtype=torch.float32, device=device)
    maxima = torch.zeros((32, 12, 16), dtype=torch.float32, device=device)
    sums = torch.ones_like(maxima)
    lengths = torch.full((32,), 256, dtype=torch.int32, device=device)
    cu = torch.arange(33, dtype=torch.int32, device=device)
    # Runtime table pitch is not used to address staging. Compile its three
    # Triton integer specializations without touching any model/cache storage.
    for pitch in (1, 8, 16):
        result = target[(32, 12)](
            output_ptr=out, segm_output_ptr=staging, segm_max_ptr=maxima,
            segm_expsum_ptr=sums, seq_lens_ptr=lengths, num_seqs=32,
            num_query_heads=12, out_scale_inv=1.0, output_stride_0=3072,
            output_stride_1=256, block_table_stride=pitch, TILE_SIZE=16,
            HEAD_SIZE=256, HEAD_SIZE_PADDED=256, query_start_len_ptr=cu,
            BLOCK_Q=2, NUM_SEGMENTS_PER_SEQ=16, USE_FP8=False)
        _remember(result)
    _WARMED.add(device)
    _STATS['warmups'][str(device)] = 3


def prepare(model):
    if not _ENABLED:
        return
    import torch
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    for layer in model.modules():
        impl = getattr(layer, 'impl', None)
        name = getattr(layer, 'layer_name', '')
        index = name.split('.layers.')[-1].split('.')[0]
        scale = getattr(layer, '_q_scale', None)
        if (not isinstance(impl, TritonAttentionImpl) or '.layers.' not in name
                or not index.isdigit() or int(index) not in range(3, 64, 4)
                or (impl.num_heads, impl.num_kv_heads, impl.head_size) != (12, 2, 256)
                or scale is None or not scale.is_cuda
                or getattr(impl, '_mach_dense_fp32_attention', False)):
            continue
        _warmup(scale.device)
        original = impl.forward

        @functools.wraps(original)
        def forward(self, *args, _original=original, _owner=layer, _name=name, **kwargs):
            owner = kwargs.get('layer', args[0] if args else None)
            query = kwargs.get('query', args[1] if len(args) > 1 else None)
            metadata = kwargs.get('attn_metadata', args[5] if len(args) > 5 else None)
            admitted = (_ENABLED and owner is _owner and owner.layer_name == _name
                        and metadata is not None and metadata.max_query_len == 1
                        and metadata.num_actual_tokens == 32 and metadata.seq_lens.shape == (32,)
                        and query is not None and query.dtype == torch.bfloat16
                        and query.is_cuda and query.shape[0] == 32 and query.numel() == 32 * 12 * 256)
            previous = getattr(_CONTEXT, 'prefix', None)
            if not admitted and previous is None:
                return _original(*args, **kwargs)
            _CONTEXT.prefix = _name if admitted else None
            try:
                return _original(*args, **kwargs)
            finally:
                _CONTEXT.prefix = previous
        impl.forward = MethodType(forward, impl)
        impl._mach_dense_fp32_attention = True
        _STATS['prepared'].append(name)


def reduction(stock, *, num_seqs, heads, kv_heads, segments, head_size, padded_head_size, bf16, use_fp8):
    prefix = getattr(_CONTEXT, 'prefix', None)
    if not (_ENABLED and prefix is not None and num_seqs == 32 and heads == 12
            and kv_heads == 2 and segments == 16 and head_size == padded_head_size == 256
            and bf16 and not use_fp8):
        return stock
    target = _target()

    class Launch:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                import torch
                values = {**dict(zip(target.arg_names, args)), **kwargs}
                expected = dict(num_seqs=32, num_query_heads=12, out_scale_inv=1.0,
                                output_stride_0=3072, output_stride_1=256, TILE_SIZE=16,
                                HEAD_SIZE=256, HEAD_SIZE_PADDED=256, BLOCK_Q=2,
                                NUM_SEGMENTS_PER_SEQ=16, USE_FP8=False)
                shapes = {'output_ptr': ((32, 12, 256), torch.bfloat16),
                          'seq_lens_ptr': ((32,), torch.int32),
                          'query_start_len_ptr': ((33,), torch.int32)}
                staging = {'segm_output_ptr': (12, 16, 256),
                           'segm_max_ptr': (12, 16), 'segm_expsum_ptr': (12, 16)}
                out = values.get('output_ptr')
                if (not isinstance(grid, (tuple, list)) or tuple(grid) != (32, 12)
                        or out is None or out.device not in _WARMED
                        or type(values.get('block_table_stride')) is not int or values['block_table_stride'] <= 0
                        or kwargs.get('num_warps', 4) != 4 or kwargs.get('num_stages', 3) != 3
                        or kwargs.get('num_ctas', 1) != 1 or kwargs.get('enable_fp_fusion', True) is not True
                        or kwargs.get('launch_pdl', False) is not False or kwargs.get('ptx_version') not in (None, 90)
                        or any(values.get(k) != v for k, v in expected.items())
                        or any(k not in values or values[k].shape != shape or values[k].dtype != dtype
                               or not values[k].is_cuda or values[k].device != out.device
                               or not values[k].is_contiguous() for k, (shape, dtype) in shapes.items())
                        # Scratch may reserve more sequences than this M32
                        # launch. Only the first32 dense rows are addressed;
                        # allocation capacity is not a JIT or arithmetic key.
                        or any(k not in values or len(values[k].shape) != len(tail) + 1
                               or values[k].shape[0] < 32 or values[k].shape[1:] != tail
                               or values[k].dtype != torch.float32 or not values[k].is_cuda
                               or values[k].device != out.device or not values[k].is_contiguous()
                               for k, tail in staging.items())):
                    return stock[grid](*args, **kwargs)
                result = target[grid](*args, **kwargs)
                _STATS['calls'][prefix] = _STATS['calls'].get(prefix, 0) + 1
                if torch.cuda.is_current_stream_capturing():
                    _STATS['capture_calls'][prefix] = _STATS['capture_calls'].get(prefix, 0) + 1
                _remember(result)
                return result
            return run
    return Launch()


def receipt():
    return dict(enabled=_ENABLED, source_verified=_VERIFIED, **_STATS,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Native Dense FP32 SSM M32 decode Hq12/KV2/D256 NT16 only',
                counters='Actual host launches/captures; warmups separate, not graph replays')
