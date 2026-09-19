# SPDX-License-Identifier: Apache-2.0
"""Capability-gated TP2 decode SwiGLU/MXFP8 producer using the extension's CUDA op."""
from __future__ import annotations

import os
import types

import torch

from .dense import Mxfp6Sm120LinearKernel, _import_mxfp6

_REGISTERED = False
_ROWS = (1, 2, 4, 8, 16, 24, 32)


def _fused_down(gate_up: torch.Tensor, weight: torch.Tensor,
                weight_scale: torch.Tensor) -> torch.Tensor:
    return torch.ops.mxfp6.gemm_from_swiglu(
        gate_up, weight, weight_scale, weight.shape[0], 1.0, gate_up.dtype)


def _fake_down(gate_up: torch.Tensor, weight: torch.Tensor,
               weight_scale: torch.Tensor) -> torch.Tensor:
    return gate_up.new_empty((gate_up.shape[0], weight.shape[0]))


def _register():
    global _REGISTERED
    if not _REGISTERED:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(op_name='mach_swiglu_mxfp6_down', op_func=_fused_down,
                                  mutates_args=[], fake_impl=_fake_down)
        _REGISTERED = True


def _eligible(module):
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.model_executor.layers.activation import SiluAndMul
    if type(module) is not Qwen2MoeMLP or module.expert_gate is not None:
        return False
    if type(module.act_fn) is not SiluAndMul:
        return False
    up, down = module.gate_up_proj, module.down_proj
    if type(up) is not MergedColumnParallelLinear or type(down) is not RowParallelLinear:
        return False
    return (up.tp_size == down.tp_size == 2 and down.input_is_parallel
            and not down.reduce_results and up.bias is None and down.bias is None
            and tuple(up.weight.shape) == (17408, 3840)
            and tuple(down.weight.shape) == (5120, 6528)
            and all(isinstance(getattr(getattr(layer, 'scheme', None), 'ocp_mx_linear', None),
                               Mxfp6Sm120LinearKernel) for layer in (up, down)))


def prepare(model):
    """Install only on native, bias-free Qwen TP2 partial-output MLPs."""
    mode = os.environ.get('VLLM_MACH_FUSED_SWIGLU_QUANT', 'auto')
    if mode == '0':
        return 0
    if mode not in ('auto', '1'):
        raise ValueError('VLLM_MACH_FUSED_SWIGLU_QUANT must be auto, 0 or 1')
    if not any(_eligible(module) for module in model.modules()):
        return 0
    extension = _import_mxfp6()
    extension.load_library()
    if not hasattr(torch.ops.mxfp6, 'gemm_from_swiglu'):
        raise RuntimeError('Fused SwiGLU requires the TP2 extension with gemm_from_swiglu')
    _register()
    count = 0
    for module in model.modules():
        if getattr(module, '_mach_swiglu_prepared', False) or not _eligible(module):
            continue
        original = module.forward

        def forward(this, x, _original=original):
            if (x.ndim != 2 or x.shape[0] not in _ROWS or x.shape[1] != 5120
                    or x.dtype != torch.bfloat16 or not x.is_cuda):
                return _original(x)
            gate_up, _ = this.gate_up_proj(x)
            return torch.ops.vllm.mach_swiglu_mxfp6_down(
                gate_up, this.down_proj.weight, this.down_proj.weight_scale)

        module.forward = types.MethodType(forward, module)
        module._mach_swiglu_prepared = True
        count += 1
    if count:
        from vllm.logger import init_logger
        init_logger("vllm.mach.mlp").info("Mach prepared %d TP2 SwiGLU/MXFP8 MLPs", count)
    return count
