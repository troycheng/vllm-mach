# SPDX-License-Identifier: Apache-2.0
"""Qualified Dense FP32 overlap decode; persistent and FP16 paths stay intact."""
import ast
import hashlib
import os
from pathlib import Path

_ROWS = (16, 24, 32)
_KERNEL = None
_WARMED = set()
_STATS = {'prepared': [], 'calls': {}, 'capture_calls': {}, 'kernels': {}, 'warmups': {}}


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from vllm.third_party.flash_linear_attention.ops import fused_recurrent, op
        from ...mxfp8.two_b.gdn.packed_fp32 import _kernel as qualified_kernel

        sources = ((fused_recurrent, 'b065f160e94371192ce743bcd1ce3a9c75ddcecf31be243af4d55b39a898685e'),
                   (op, '456da84fd12411cf53c96a90dd4f78f49afb454c3d51d2c524ea8e7106fe8ae5'))
        for module, expected in sources:
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Dense packed source changed: ' + module.__name__)
        name = 'fused_recurrent_gated_delta_rule_packed_decode_kernel'
        node = ast.parse(getattr(fused_recurrent, name).src).body[0]
        expected = next(n for n in ast.parse(Path(fused_recurrent.__file__).read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        node.decorator_list = expected.decorator_list = []
        if ast.dump(node) != ast.dump(expected):
            raise RuntimeError('Dense packed JIT source changed')
        _KERNEL = qualified_kernel()
    return _KERNEL


def _launch(mixed, a, b, A_log, dt_bias, state, out, indices):
    return _kernel()[(4, mixed.shape[0] * 24)](
        mixed, a, b, A_log, dt_bias, out, state, state, indices, 128**-0.5,
        stride_mixed_qkv_tok=8192, stride_a_tok=24, stride_b_tok=24,
        stride_init_state_token=401408, stride_final_state_token=401408,
        stride_indices_seq=1, H=8, HV=24, K=128, V=128, BK=128, BV=32,
        SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=True,
        SPLIT_BATCH_HEAD_GRID=False, num_warps=1, num_stages=3, ptx_version=90)


def _remember(result):
    if result.hash not in _STATS['kernels']:
        _STATS['kernels'][result.hash] = dict(
            num_warps=result.metadata.num_warps, num_stages=result.metadata.num_stages,
            ptx_version=result.metadata.ptx_version,
            ptx_sha256=hashlib.sha256(result.asm['ptx'].encode()).hexdigest())


def prepare(layer):
    if (getattr(layer, '_mach_dense_fp32_packed', False)
            or not getattr(layer, '_mach_dense_norm', False)
            or not getattr(layer, '_mach_gdn_prepared', False)
            or os.environ.get('FLA_USE_FAST_OPS', '0') != '0'):
        return
    import torch

    if (layer.get_state_dtype()[1] != torch.float32
            or layer.gdn_prefill_backend != 'triton'
            or layer.tp_size != 2 or layer.num_k_heads != 16 or layer.num_v_heads != 48):
        return
    device = layer.A_log.device
    if device not in _WARMED:
        # No kv_cache dependency: cache allocation occurs later in this lifecycle.
        mixed = torch.empty_strided((32, 5120), (8192, 1), dtype=torch.bfloat16, device=device)
        ab = torch.zeros((32, 24), dtype=torch.bfloat16, device=device)
        state = torch.empty_strided((2, 24, 128, 128), (401408, 16384, 128, 1), dtype=torch.float32, device=device)
        out = torch.empty((32, 1, 24, 128), dtype=torch.bfloat16, device=device)
        indices = torch.full((32,), -1, dtype=torch.int32, device=device)
        _remember(_launch(mixed, ab, ab, layer.A_log, layer.dt_bias, state, out, indices))
        _WARMED.add(device)
        _STATS['warmups'][str(device)] = 1
    layer._mach_dense_fp32_packed = True
    _STATS['prepared'].append(layer.prefix)


def decode(layer, mixed, a, b, state, out, indices):
    if not getattr(layer, '_mach_dense_fp32_packed', False):
        return False
    import torch

    rows = mixed.shape[0]
    tensors = (mixed, a, b, layer.A_log, layer.dt_bias, state, out, indices)
    if (rows not in _ROWS or mixed.shape != (rows, 5120)
            or mixed.stride() != (8192, 1)
            or a.shape != b.shape or a.shape != (rows, 24)
            or a.stride() != b.stride() or a.stride() != (24, 1)
            or layer.A_log.shape != (24,) or layer.A_log.stride() != (1,)
            or layer.dt_bias.shape != (24,) or layer.dt_bias.stride() != (1,)
            or state.ndim != 4 or state.shape[1:] != (24, 128, 128)
            or state.dtype != torch.float32 or state.stride() != (401408, 16384, 128, 1)
            or out.shape != (rows, 1, 24, 128) or not out.is_contiguous()
            or indices.shape != (rows,) or indices.stride() != (1,)
            or indices.dtype != torch.int32
            or layer.A_log.dtype != torch.float32
            or any(t.dtype != torch.bfloat16 for t in (mixed, a, b, layer.dt_bias, out))
            or not all(t.is_cuda and t.device == mixed.device for t in tensors)):
        return False
    result = _launch(mixed, a, b, layer.A_log, layer.dt_bias, state, out, indices)
    key = layer.prefix + ':m' + str(rows)
    _STATS['calls'][key] = _STATS['calls'].get(key, 0) + 1
    if torch.cuda.is_current_stream_capturing():
        _STATS['capture_calls'][key] = _STATS['capture_calls'].get(key, 0) + 1
    _remember(result)
    return True


def receipt():
    root = Path(__file__).parents[2] / 'mxfp8'
    return dict(**_STATS,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                kernel_builder_sha256=hashlib.sha256((root / 'two_b/gdn/packed_fp32.py').read_bytes()).hexdigest(),
                reduction_helper_sha256=hashlib.sha256((root / 'gdn/legacy_reduction.py').read_bytes()).hexdigest(),
                scope='Prepared Dense FP32 overlap M16/24/32 only; native M4 persistent and all FP16 paths unchanged',
                counters='Completed target host launches/captures; dead-buffer warmup is separate, not graph replays')
