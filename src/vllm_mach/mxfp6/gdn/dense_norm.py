# SPDX-License-Identifier: Apache-2.0
"""Original R128 norm reductions on prepared native Dense instances."""
import functools
import hashlib
import threading
from pathlib import Path

from . import dense_prep

_GUARD_SHA256 = 'b4f696187a1326f9aac44f23d0bb5db814dc28ebe4fc31da000edc24365e09c6'
_CONTEXT = threading.local()
_INSTALLED = False
_WARMED = set()
_STATS = {'prepared': [], 'calls': {}, 'capture_calls': {}, 'kernels': {}, 'warmups': {}}


def _remember(kernel):
    key = kernel.hash
    if key not in _STATS['kernels']:
        _STATS['kernels'][key] = dict(
            num_warps=kernel.metadata.num_warps,
            num_stages=kernel.metadata.num_stages,
            ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest())
    return key


def _install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as guard
    from ...fp8._guard_kernel import layer_norm_fwd_kernel

    if hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest() != _GUARD_SHA256:
        raise RuntimeError('Dense norm guard source changed')
    original = guard.layer_norm_fwd_kernel.run

    @functools.wraps(original)
    def run(*args, **kwargs):
        prefix = getattr(_CONTEXT, 'prefix', None)
        expected = dict(BLOCK_N=128, HAS_BIAS=False, HAS_Z=True,
                        IS_RMS_NORM=True, NORM_BEFORE_GATE=True,
                        ACTIVATION='silu', num_warps=1)
        if (prefix is None or len(args) != 13 or args[11] != 128
                or args[7:10] != (128, 128, 128) or args[12] != 1e-6
                or kwargs.get('ROWS_PER_BLOCK') not in (1, 2, 4)
                or any(kwargs.get(k) != v for k, v in expected.items())):
            return original(*args, **kwargs)
        result = layer_norm_fwd_kernel.run(*args, **kwargs)
        key = prefix + ':row' + str(kwargs['ROWS_PER_BLOCK']) + ':m' + str(args[10])
        _STATS['calls'][key] = _STATS['calls'].get(key, 0) + 1
        if torch.cuda.is_current_stream_capturing():
            _STATS['capture_calls'][key] = _STATS['capture_calls'].get(key, 0) + 1
        _remember(result)
        return result

    guard.layer_norm_fwd_kernel.run = run
    _INSTALLED = True


def _warmup(layer):
    """Compile both M alignments and all supported row keys on dead buffers."""
    import torch
    from ...fp8._guard_kernel import layer_norm_fwd_kernel

    device = layer.norm.weight.device
    if device in _WARMED:
        return
    for rows in (24, 48):
        x = torch.zeros((rows, 128), dtype=torch.bfloat16, device=device)
        z, out = torch.zeros_like(x), torch.empty_like(x)
        rstd = torch.empty(rows, dtype=torch.float32, device=device)
        for block_rows in (1, 2, 4):
            result = layer_norm_fwd_kernel[(lambda _: ((rows + block_rows - 1) // block_rows, 1))](
                x, out, layer.norm.weight, None, z, None, rstd,
                128, 128, 128, rows, 128, 1e-6, BLOCK_N=128,
                ROWS_PER_BLOCK=block_rows, NORM_BEFORE_GATE=True,
                IS_RMS_NORM=True, ACTIVATION='silu', num_warps=1)
            _remember(result)
    _STATS['warmups'][str(device)] = 6
    _WARMED.add(device)


def _operands(x, z):
    import torch

    return (z is not None and x.ndim == 3 and x.shape[0] > 0
            and x.shape[1:] == (24, 128) and x.shape == z.shape
            and x.stride() == (3072, 128, 1)
            and z.stride() in ((3072, 128, 1), (8192, 128, 1))
            and x.dtype == z.dtype == torch.bfloat16
            and x.is_cuda and z.device == x.device)


def prepare(layer):
    """Keep the original frontend and direct OWNER/decode output contract."""
    if (getattr(layer, '_mach_dense_norm', False)
            or not dense_prep._ENABLED
            or not getattr(layer, '_mach_gdn_prepared', False)):
        return
    import torch
    from vllm.model_executor.layers.layernorm import RMSNormGated

    if (not getattr(layer, '_mach_gdn_prepared', False)
            or not dense_prep.model_eligible(layer.vllm_config)
            or not dense_prep._sm120() or layer.tp_size != 2
            or layer.num_k_heads != 16 or layer.num_v_heads != 48
            or layer.get_state_dtype()[1] not in (torch.float32, torch.float16)):
        return
    norm = layer.norm
    if (type(norm) is not RMSNormGated or norm.bias is not None
            or norm.weight.shape != (128,) or norm.weight.dtype != torch.bfloat16
            or norm.eps != 1e-6 or norm.group_size is not None
            or norm.norm_before_gate is not True or norm.activation != 'silu'
            or getattr(norm._forward_method, '__func__', None) is not RMSNormGated.forward_cuda):
        return
    _install()
    original, original_direct = norm._forward_method, layer._rms_norm_gated_cuda

    def invoke(fn, *args):
        previous = getattr(_CONTEXT, 'prefix', None)
        _CONTEXT.prefix = layer.prefix
        try:
            return fn(*args)
        finally:
            _CONTEXT.prefix = previous

    @functools.wraps(original)
    def forward(x, z=None):
        if not _operands(x, z):
            return original(x, z)
        return invoke(original, x, z)

    @functools.wraps(original_direct)
    def direct(x, output_gate, out):
        if (not _operands(x, output_gate) or out.shape != x.shape
                or out.stride() != x.stride() or out.dtype != x.dtype
                or out.device != x.device):
            return original_direct(x, output_gate, out)
        return invoke(original_direct, x, output_gate, out)

    # Warm before installing the instance routes and before FULL graph capture.
    _warmup(layer)
    norm._forward_method = forward
    layer._rms_norm_gated_cuda = direct
    layer._mach_dense_norm = True
    _STATS['prepared'].append(layer.prefix)


def receipt():
    helper = Path(__file__).parents[2] / 'fp8' / '_guard_kernel.py'
    return dict(installed=_INSTALLED, **_STATS,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                scope='Prepared native Dense H8/HV24, FP32/FP16 SSM, BF16 R128 norm only; original backend/frontend preserved',
                counters='Completed host norm calls/captures, excluding dead-buffer warmup and graph replays')
