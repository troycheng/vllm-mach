# SPDX-License-Identifier: Apache-2.0
"""Fuse Qwen3.5 TP1 SwiGLU and MXFP8 activation quantization."""
from __future__ import annotations

import os
import types

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from vllm.utils.torch_utils import direct_register_custom_op


def _down(gate_up: torch.Tensor, weight: torch.Tensor,
          weight_scale: torch.Tensor) -> torch.Tensor:
    if gate_up.shape[0] == 0:
        return gate_up.new_empty((0, weight.shape[0]))
    a, scales = _triton_quantize(gate_up)
    from . import dense_mxfp8

    return dense_mxfp8._packed_gemm(a, weight, scales, weight_scale)


def _fake(gate_up, weight, weight_scale):
    return gate_up.new_empty((gate_up.shape[0], weight.shape[0]))


direct_register_custom_op(op_name="mach_swiglu_mxfp8_down", op_func=_down,
                          mutates_args=[], fake_impl=_fake)


def _eligible(module):
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    from .dense_mxfp8 import Mxfp8Sm120LinearKernel

    if type(module) is not Qwen2MoeMLP or module.expert_gate is not None:
        return False
    up, down = module.gate_up_proj, module.down_proj
    return (
        type(module.act_fn) is SiluAndMul
        and type(up) is MergedColumnParallelLinear
        and type(down) is RowParallelLinear
        and up.tp_size == down.tp_size == 1 and down.input_is_parallel
        and up.bias is None and down.bias is None
        and tuple(up.weight.shape) == (18432, 2560)
        and tuple(down.weight.shape) == (2560, 9216)
        and up.weight.is_cuda and down.weight.is_cuda
        and all(isinstance(getattr(getattr(layer, "scheme", None), "kernel", None),
                           Mxfp8Sm120LinearKernel) for layer in (up, down))
    )


def prepare(model):
    if os.environ.get("VLLM_MACH_MXFP8_FUSED_MLP", "1") != "1":
        return 0
    count = 0
    for module in model.modules():
        if getattr(module, "_mach_mxfp8_mlp", False) or not _eligible(module):
            continue
        original = module.forward

        def forward(this, x, _original=original):
            if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
                return _original(x)
            gate_up, _ = this.gate_up_proj(x)
            return torch.ops.vllm.mach_swiglu_mxfp8_down(
                gate_up, this.down_proj.weight, this.down_proj.weight_scale)

        module.forward = types.MethodType(forward, module)
        module._mach_mxfp8_mlp = True
        count += 1
    if count:
        from vllm.logger import init_logger
        init_logger(__name__).info("Mach fused SwiGLU/MXFP8 enabled for %d MLPs", count)
    return count


@triton.jit
def _silu_quant(X, Y, S, K: tl.constexpr, GROUPS: tl.constexpr,
                PDL: tl.constexpr):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    row = tl.program_id(0)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    cols = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    gate = tl.load(X + row * 2 * K + cols, cols < K, 0).to(tl.float32)
    up = tl.load(X + row * 2 * K + K + cols, cols < K, 0).to(tl.float32)
    act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    values = (act * up).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(values), 1)
    raw = tl.maximum(amax / 448.0, 1.0e-30)
    bits = (raw.to(tl.uint32, bitcast=True) + 0x007fffff) & 0x7f800000
    scale = bits >> 23
    inverse = 1.0 / bits.to(tl.float32, bitcast=True)
    quant = (values * inverse[:, None]).to(tl.float8e4nv)
    tl.store(Y + row * K + cols, quant, cols < K)
    num_k_tiles: tl.constexpr = (K // 32 + 3) // 4
    offset = (((row // 128 * num_k_tiles + groups // 4) * 32 + row % 32)
              * 4 + row % 128 // 32) * 4 + groups % 4
    tl.store(S + offset, scale.to(tl.uint8), groups < K // 32)


def _triton_quantize(x, groups=None):
    from .dense_mxfp8 import _backend, _pdl_enabled

    m, two_k = x.shape
    pdl = _backend() == "native" and _pdl_enabled(m)
    if groups is None:
        groups = 8 if m <= 4 else 32
    k = two_k // 2
    values = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((triton.cdiv(m, 128) * 128 * triton.cdiv(k//32, 4) * 4,),
                         device=x.device, dtype=torch.uint8)
    if m:
        _silu_quant[(m, triton.cdiv(k//32, groups))](
            x, values, scales, k, groups, pdl, num_warps=4, launch_pdl=pdl)
    return values, scales
