# SPDX-License-Identifier: Apache-2.0
"""Direct output caller at the qualified Block H32/R128 norm boundary."""
import torch
import triton
from vllm.third_party.flash_linear_attention.ops import layernorm_guard as guard
from ._guard_direct_kernel import layer_norm_fwd_kernel


def run(x, weight, z, out):
    assert x.shape == z.shape == out.shape and x.shape[1:] == (32, 128)
    assert x.stride() == out.stride() == (4096, 128, 1)
    assert z.stride() == (12288, 128, 1)
    assert x.dtype == weight.dtype == z.dtype == out.dtype == torch.bfloat16
    assert weight.shape == (128,) and weight.stride() == (1,)
    m = x.shape[0] * 32
    rows = guard.calc_rows_per_block(m, x.device)
    assert rows in (1, 2, 4)
    rstd = torch.empty((m,), dtype=torch.float32, device=x.device)
    layer_norm_fwd_kernel[(triton.cdiv(m, rows), 1)](
        x, out, weight, None, z, None, rstd,
        128, 128, 128, m, 128, 1e-6,
        BLOCK_N=128, ROWS_PER_BLOCK=rows, HAS_BIAS=False, HAS_Z=True,
        NORM_BEFORE_GATE=True, IS_RMS_NORM=True, ACTIVATION='silu',
        num_warps=1, num_stages=3)
    return out
