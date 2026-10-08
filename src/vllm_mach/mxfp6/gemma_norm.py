# SPDX-License-Identifier: Apache-2.0
"""Fuse Gemma's FP32 residual, weight offset and RMS normalization on SM120."""
import os
from functools import partial

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _norm(X, R, W, Y, S, N: tl.constexpr, EPS: tl.constexpr,
          RESIDUAL: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    weight = tl.load(W + cols, cols < N, 0).to(tl.float32) + 1.0
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    x = tl.load(X + row * N + cols, cols < N, 0).to(tl.float32)
    if RESIDUAL:
        x = x + tl.load(R + row * N + cols, cols < N, 0).to(tl.float32)
        # Preserve the BF16 residual output, but normalize the unrounded FP32 sum.
        tl.store(S + row * N + cols, x, cols < N)
    variance = tl.sum(x * x, 0) / N
    y = (x * tl.rsqrt(variance + EPS)) * weight
    tl.store(Y + row * N + cols, y, cols < N)


def _impl(x: torch.Tensor, residual: torch.Tensor | None,
          weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.empty_like(x)
    summed = torch.empty_like(x) if residual is not None else x.new_empty((0,))
    n = x.shape[-1]
    if x.numel():
        from .dense_mxfp8 import _backend, _pdl_enabled

        pdl = _backend() == "native" and _pdl_enabled(x.numel() // n)
        _norm[(x.numel() // n,)](x, residual, weight, y, summed, n, eps,
                                residual is not None, triton.next_power_of_2(n),
                                pdl, num_warps=4, enable_fp_fusion=False,
                                launch_pdl=pdl)
    return y, summed


def _fake(x, residual, weight, eps: float):
    return torch.empty_like(x), (
        torch.empty_like(x) if residual is not None else x.new_empty((0,))
    )


direct_register_custom_op(op_name="mach_gemma_norm", op_func=_impl,
                          mutates_args=[], fake_impl=_fake)


def _forward(norm, fallback, x, residual=None):
    if (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.device == norm.weight.device
            and x.shape[-1] == norm.weight.numel()
            and (residual is None or (residual.shape == x.shape
                 and residual.dtype == x.dtype and residual.device == x.device
                 and residual.is_contiguous()))):
        y, summed = torch.ops.vllm.mach_gemma_norm(
            x, residual, norm.weight, norm.variance_epsilon)
        return y if residual is None else (y, summed)
    return fallback(x, residual)


def prepare(model):
    """Install only on native MXFP8 TP1 models; leave TP2 producers in control."""
    if os.getenv("VLLM_MACH_FUSED_GEMMA_NORM", "1") == "0":
        return
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    from .dense_mxfp8 import Mxfp8Sm120LinearKernel

    if not any(isinstance(getattr(getattr(m, "scheme", None), "kernel", None),
                          Mxfp8Sm120LinearKernel) for m in model.modules()):
        return
    if get_tensor_model_parallel_world_size() != 1:
        return
    count = 0
    for norm in model.modules():
        if (type(norm) is not GemmaRMSNorm or hasattr(norm, "_mach_gemma_norm")
                or not norm.weight.is_cuda or not norm.weight.is_contiguous()
                or norm.weight.dtype != torch.bfloat16
                or norm.weight.numel() > 8192
                or torch.cuda.get_device_capability(norm.weight.device) != (12, 0)):
            continue
        norm._forward_method = partial(_forward, norm, norm._forward_method)
        norm._mach_gemma_norm = True
        count += 1
    if count:
        from vllm.logger import init_logger
        init_logger(__name__).info("Mach fused GemmaRMSNorm enabled for %d norms", count)
