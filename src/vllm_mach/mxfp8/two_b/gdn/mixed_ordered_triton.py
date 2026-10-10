# SPDX-License-Identifier: Apache-2.0
"""Mixed-decode W4 kernel with stock mixed-branch FP32 operation order.

Pure decode and mixed decode have different stock normalizations. Keep this
body separate to preserve their respective rounding; the scratch layout and
materialization implementation are shared.
"""
from vllm.triton_utils import tl, triton
from .ordered_allm_triton import HV, H, V, K, BV, NV, W, validate


@triton.jit
def _ordered_decode(mixed_qkv, a, b, A_log, dt_bias, out, base, indices,
                    pending_k, pending_d, coeff, prefix, age,
                    stride_mixed: tl.constexpr, stride_a: tl.constexpr,
                    stride_b: tl.constexpr, stride_base_page: tl.constexpr,
                    stride_index: tl.constexpr, P: tl.constexpr, W: tl.constexpr,
                    SCALE: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                    V: tl.constexpr, K: tl.constexpr, BV: tl.constexpr,
                    NV: tl.constexpr):
    iv = tl.program_id(0)
    ihv = tl.program_id(1)
    it = tl.program_id(2)
    ik = ihv // (HV // H)
    ok = tl.arange(0, K)
    ov_local = tl.arange(0, BV)
    ov = iv * BV + ov_local
    state_idx = tl.load(indices + it * stride_index).to(tl.int32)
    output_ptr = out + (it * HV + ihv) * V + ov
    if (state_idx <= 0) | (state_idx >= P):
        tl.store(output_ptr, tl.full((BV,), 0, tl.float32).to(out.dtype.element_ty))
        return

    meta = (state_idx * HV + ihv) * NV + iv
    old_age = tl.load(age + meta)
    if (old_age < 0) | (old_age >= W):
        tl.store(output_ptr, tl.full((BV,), 0, tl.float32).to(out.dtype.element_ty))
        return
    ptr_base = base + state_idx * stride_base_page + ihv * V * K + ov[:, None] * K + ok[None, :]
    b_h = tl.load(ptr_base).to(tl.float32)
    pk_head = pending_k + (state_idx * HV + ihv) * W * K
    pd_tile = pending_d + meta * W * BV
    pc_tile = coeff + meta * W
    # Replay the *stored raw* factors in token order. In particular, multiply
    # dense H before the rank update; no prefix/product or projection shortcut.
    for j in range(W):
        if j < old_age:
            old_alpha = tl.load(pc_tile + j).to(tl.float32)
            old_k = tl.load(pk_head + j * K + ok).to(tl.float32)
            old_d = tl.load(pd_tile + j * BV + ov_local).to(tl.float32)
            # Stock consumes rounded H*alpha in its dot product before the
            # update. Pin the same FMA operand order for replay/materialize.
            scaled = b_h * old_alpha
            b_h = tl.fma(old_d[:, None], old_k[None, :], scaled)

    ptr_mixed = mixed_qkv + it * stride_mixed
    b_q = tl.load(ptr_mixed + ik * K + ok).to(tl.float32)
    b_k = tl.load(ptr_mixed + H * K + ik * K + ok).to(tl.float32)
    b_v = tl.load(ptr_mixed + 2 * H * K + ihv * V + ov).to(tl.float32)
    b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
    b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
    b_q = b_q * SCALE
    a_val = tl.load(a + it * stride_a + ihv).to(tl.float32)
    b_val = tl.load(b + it * stride_b + ihv).to(tl.float32)
    A_log_val = tl.load(A_log + ihv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + ihv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val)

    # Keep the stock packed FLA update/reduction source expressions and order.
    alpha = tl.exp(g_val)
    b_h *= alpha
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(output_ptr, b_o.to(out.dtype.element_ty))

    if old_age == W - 1:
        tl.store(ptr_base, b_h)
        tl.store(age + meta, 0)
        tl.store(prefix + meta, 1.0)
    else:
        if iv == 0:
            tl.store(pk_head + old_age * K + ok, b_k)
        tl.store(pd_tile + old_age * BV + ov_local, b_v)
        tl.store(pc_tile + old_age, alpha)
        tl.store(age + meta, old_age + 1)
