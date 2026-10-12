# SPDX-License-Identifier: Apache-2.0
"""Optional TP1 residual/GemmaRMSNorm/quantization producer for dense MLPs."""
from __future__ import annotations

import os
import types

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _norm_quant(X, R, W, A, S, OUT_R, N: tl.constexpr, EPS: tl.constexpr,
                BLOCK: tl.constexpr, PDL: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    weight = tl.load(W + cols, cols < N, 0).to(tl.float32) + 1.0
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    x = (tl.load(X + row * N + cols, cols < N, 0).to(tl.float32)
         + tl.load(R + row * N + cols, cols < N, 0).to(tl.float32))
    tl.store(OUT_R + row * N + cols, x, cols < N)
    variance = tl.sum(x * x, 0) / N
    y = ((x * tl.rsqrt(variance + EPS)) * weight).to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(y, (BLOCK // 32, 32))
    raw = tl.maximum(tl.max(tl.abs(grouped), 1) / 448.0, 1.0e-30)
    bits = (raw.to(tl.uint32, bitcast=True) + 0x007fffff) & 0x7f800000
    inverse = 1.0 / bits.to(tl.float32, bitcast=True)
    quant = (grouped * inverse[:, None]).to(tl.float8e4nv)
    tl.store(A + row * N + cols, tl.reshape(quant, (BLOCK,)), cols < N)
    groups = tl.arange(0, BLOCK // 32)
    tiles: tl.constexpr = triton.cdiv(N // 32, 4)
    offset = (((row // 128 * tiles + groups // 4) * 32 + row % 32)
              * 4 + row % 128 // 32) * 4 + groups % 4
    tl.store(S + offset, (bits >> 23).to(tl.uint8), groups < N // 32)


def quantize(x, residual, weight, eps):
    from .dense_mxfp8 import _backend, _pdl_enabled

    m, n = x.shape
    a = torch.empty((m, n), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((triton.cdiv(m, 128) * 128 * triton.cdiv(n // 32, 4) * 4,),
                         device=x.device, dtype=torch.uint8)
    summed = torch.empty_like(x)
    if m:
        pdl = _backend() == "native" and _pdl_enabled(m)
        _norm_quant[(m,)](x, residual, weight, a, scales, summed, n, eps,
                          triton.next_power_of_2(n), pdl, num_warps=4,
                          enable_fp_fusion=False, launch_pdl=pdl)
    return a, scales, summed


def _gate_up(x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
             eps: float, weight: torch.Tensor,
             weight_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from . import dense_mxfp8

    if x.shape[0] > 32:
        from .gemma_norm import _impl

        norm, summed = _impl(x, residual, norm_weight, eps)
        return dense_mxfp8._gemm(norm, weight, weight_scale), summed
    a, scales, summed = quantize(x, residual, norm_weight, eps)
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[0])), summed
    y = dense_mxfp8._packed_gemm(a, weight, scales, weight_scale)
    return y, summed


def _fake(x, residual, norm_weight, eps: float, weight, weight_scale):
    return x.new_empty((x.shape[0], weight.shape[0])), torch.empty_like(x)


direct_register_custom_op(op_name="mach_norm_quant_mxfp8_gate_up", op_func=_gate_up,
                          mutates_args=[], fake_impl=_fake)


def prepare(model):
    if os.getenv("VLLM_MACH_MXFP8_NORM_QUANT", "0") != "1":
        return 0
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    from .mxfp8_mlp import _eligible

    count = 0
    for layer in model.modules():
        if (type(layer) is not Qwen3_5DecoderLayer
                or getattr(layer, "_mach_norm_quant", False)
                or layer.layer_scale or layer.use_fused_ar_gemma_norm
                or layer.use_attn_reduce_scatter_for_moe
                or not _eligible(layer.mlp)
                or not getattr(layer.mlp, "_mach_mxfp8_mlp", False)
                or not getattr(layer.post_attention_layernorm, "_mach_gemma_norm", False)):
            continue
        original = layer.forward

        def forward(this, hidden_states, residual, positions=None, _original=original,
                    **kwargs):
            if (hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16
                    or not hidden_states.is_contiguous()):
                return _original(hidden_states, residual, positions, **kwargs)
            if residual is None:
                residual = hidden_states
                hidden_states = this.input_layernorm(hidden_states)
            else:
                hidden_states, residual = this.input_layernorm(hidden_states, residual)
            if this.layer_type == "linear_attention":
                hidden_states = this.linear_attn(hidden_states=hidden_states)
            else:
                hidden_states = this.self_attn(hidden_states=hidden_states, positions=positions)
            up = this.mlp.gate_up_proj
            norm = this.post_attention_layernorm
            gate_up, residual = torch.ops.vllm.mach_norm_quant_mxfp8_gate_up(
                hidden_states, residual, norm.weight, norm.variance_epsilon,
                up.weight, up.weight_scale)
            down = this.mlp.down_proj
            return torch.ops.vllm.mach_swiglu_mxfp8_down(
                gate_up, down.weight, down.weight_scale), residual

        layer.forward = types.MethodType(forward, layer)
        layer._mach_norm_quant = True
        count += 1
    if count:
        from vllm.logger import init_logger
        init_logger(__name__).info("Mach residual/GemmaRMSNorm/MXFP8 producer enabled for %d MLPs", count)
    return count
