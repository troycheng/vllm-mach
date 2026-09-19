# SPDX-License-Identifier: Apache-2.0
"""Capability-gated TP2 GDN norm/MXFP8 producer owned by mxfp6-sm120."""
from __future__ import annotations

import os
import torch
from .dense import Mxfp6Sm120LinearKernel, _import_mxfp6

_REGISTERED = False


def _project(core: torch.Tensor, gate: torch.Tensor, norm_weight: torch.Tensor,
             weight: torch.Tensor, weight_scale: torch.Tensor, eps: float) -> torch.Tensor:
    return _import_mxfp6().gemm_from_gdn(core, gate, norm_weight, weight, weight_scale, eps)


def _fake(core, gate, norm_weight, weight, weight_scale, eps: float):
    return core.new_empty((core.shape[0], weight.shape[0]))


def _eligible(layer):
    norm, out = layer.norm, layer.out_proj
    return (norm.bias is None and norm.norm_before_gate
            and norm.activation in ('silu', 'swish')
            and norm.group_size in (None, 128)
            and tuple(norm.weight.shape) == (128,)
            and norm.weight.dtype in (torch.bfloat16, torch.float32)
            and out.tp_size == 2 and out.input_is_parallel and not out.reduce_results
            and out.bias is None and tuple(out.weight.shape) == (5120, 2304)
            and isinstance(getattr(getattr(out, 'scheme', None), 'ocp_mx_linear', None),
                           Mxfp6Sm120LinearKernel))


def prepare(layer):
    mode = os.environ.get('VLLM_MACH_FUSED_GDN_QUANT', 'auto')
    if mode not in ('auto', '0', '1'):
        raise ValueError('VLLM_MACH_FUSED_GDN_QUANT must be auto, 0 or 1')
    if mode == '0' or not _eligible(layer):
        return False
    ext = _import_mxfp6()
    ext.load_library()
    if not (hasattr(ext, 'gemm_from_gdn') and hasattr(torch.ops.mxfp6, 'gemm_w6a8_pdl')):
        raise RuntimeError('Fused GDN requires gemm_from_gdn and gemm_w6a8_pdl')
    global _REGISTERED
    if not _REGISTERED:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(op_name='mach_gdn_norm_mxfp6_out', op_func=_project,
                                  mutates_args=[], fake_impl=_fake)
        _REGISTERED = True
    layer._mach_gdn_output_fused = True
    return True


def project(layer, core, gate):
    m = core.shape[0]
    if (getattr(layer, '_mach_gdn_output_fused', False)
            and m in (1, 2, 4, 8, 16, 24, 32)
            and core.shape == (m, 24, 128) and core.is_contiguous()
            and gate.shape == (m, 3072) and gate.stride(1) == 1):
        return torch.ops.vllm.mach_gdn_norm_mxfp6_out(
            core, gate.view(m,24,128), layer.norm.weight,
            layer.out_proj.weight, layer.out_proj.weight_scale, layer.norm.eps)
    layer._rms_norm_gated_cuda(core, gate.reshape(m,24,128), core)
    return layer.out_proj(core.flatten(-2))[0]
