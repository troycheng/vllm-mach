# SPDX-License-Identifier: Apache-2.0
"""Exact two-stage NVFP4 global scale for the qualified 2B M24 activation."""
import triton
import triton.language as tl

@triton.jit
def _post(x):
    clamped = tl.maximum(x, 1e-12)
    reciprocal = tl.div_rn(1.0, clamped)
    scaled = reciprocal * 2688.0
    return tl.where(x > 0.0, scaled, 1.0)

@triton.jit
def _maximum_propagate_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

@triton.jit
def partial_absmax(X, P, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    maximum = tl.reduce(tl.abs(x), 0, _maximum_propagate_nan)
    tl.store(P + tl.program_id(0), maximum)

@triton.jit
def finish_scale(P, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    maximum = tl.reduce(tl.load(P + i, i < N, other=0), 0, _maximum_propagate_nan)
    tl.store(Y, _post(maximum))

def fused_scale(x):
    import torch
    partial = torch.empty((48,), dtype=torch.float32, device=x.device)
    output = torch.empty((), dtype=torch.float32, device=x.device)
    partial_absmax[48,](x, partial, 49152, 1024, num_warps=4, num_stages=1, enable_fp_fusion=False)
    finish_scale[1,](partial, output, 48, 64, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return output
