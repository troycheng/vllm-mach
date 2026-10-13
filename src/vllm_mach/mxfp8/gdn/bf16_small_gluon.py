# SPDX-License-Identifier: Apache-2.0
"""BF16 ordered GDN with explicit Gluon reduction layouts.

PDL + grouped GDN is NOT FULLY TUNED: incremental serving gains are unproven
and intermittent output discrepancies remain unresolved. Both flags default off.

Keep the packed stock kernel's two different reduction trees: q/k norm uses
four adjacent elements per lane, while state dots use eight. Replicate the
norm across warps and distribute independent state rows across them. Group
four state rows per thread for deferred decode to reduce address/reduction
registers without changing floating-point associations.

Scratch remains the original BV32/W4 layout, so batches can move between this
kernel, the large-batch Triton kernel, and stock materialization at any age.
"""
import os

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Resolve once, before worker warmup/capture; never change captured policy.
_PDL_SETTING = os.environ.get("VLLM_MACH_MXFP8_BF16_GDN_PDL", "0")
if _PDL_SETTING not in ("0", "1"):
    raise ValueError("VLLM_MACH_MXFP8_BF16_GDN_PDL must be 0 or 1")
_USE_PDL = _PDL_SETTING == "1"
_GROUPED_SETTING = os.environ.get("VLLM_MACH_MXFP8_BF16_GDN_GROUPED", "0")
if _GROUPED_SETTING not in ("0", "1"):
    raise ValueError("VLLM_MACH_MXFP8_BF16_GDN_GROUPED must be 0 or 1")
# Experimental until the intermittent long-run discrepancy is explained.
_USE_GROUPED = _GROUPED_SETTING == "1"


def supports_shape(rows, heads):
    return rows * heads <= 1536 or _USE_GROUPED

@gluon.jit
def _ordered_decode(mixed_qkv, a, b, A_log, dt_bias, out, base, indices,
                    pending_k, pending_d, coeff, prefix, age,
                    stride_mixed: gl.constexpr, stride_a: gl.constexpr,
                    stride_b: gl.constexpr, stride_base_page: gl.constexpr,
                    stride_index: gl.constexpr, P: gl.constexpr, W: gl.constexpr,
                    SCALE: gl.constexpr, H: gl.constexpr, HV: gl.constexpr,
                    V: gl.constexpr, K: gl.constexpr, BV: gl.constexpr,
                    NV: gl.constexpr, FULL: gl.constexpr,
                    STATE_ROWS: gl.constexpr = 1,
                    USE_PDL: gl.constexpr = False):
    if USE_PDL:
        # State, indices and pending history may also be produced upstream.
        # Wait before *any* load so repeated decode and slot resets stay safe.
        gl.inline_asm_elementwise("griddepcontrol.wait; // $0", "=r", [],
                                 dtype=gl.int32, is_pure=False, pack=1)
        gl.inline_asm_elementwise("griddepcontrol.launch_dependents; // $0",
                                 "=r", [], dtype=gl.int32, is_pure=False, pack=1)
    iv = gl.program_id(0)
    ihv = gl.program_id(1)
    it = gl.program_id(2)
    ik = ihv // (HV // H)
    state_layout: gl.constexpr = gl.BlockedLayout([STATE_ROWS, 8], [2, 16], [gl.num_warps(), 1], [1, 0])
    norm_layout: gl.constexpr = gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    ok = gl.arange(0, K, layout=gl.SliceLayout(0, state_layout))
    ov_local = gl.arange(0, BV, layout=gl.SliceLayout(1, state_layout))
    norm_k = gl.arange(0, K, layout=norm_layout)
    ov = iv * BV + ov_local
    state_idx = gl.load(indices + it * stride_index).to(gl.int32)
    output_ptr = out + (it * HV + ihv) * V + ov
    if (state_idx <= 0) | (state_idx >= P):
        gl.store(output_ptr, gl.full((BV,), 0, gl.float32, gl.SliceLayout(1, state_layout)).to(out.dtype.element_ty))
        return

    ptr_base = base + state_idx * stride_base_page + ihv * V * K + ov[:, None] * K + ok[None, :]
    b_h = gl.load(ptr_base).to(gl.float32)
    meta = (state_idx * HV + ihv) * NV + iv
    old_age = gl.load(age + meta)
    if (old_age < 0) | (old_age >= W):
        gl.store(output_ptr, gl.full((BV,), 0, gl.float32, gl.SliceLayout(1, state_layout)).to(out.dtype.element_ty))
        return
    pk_head = pending_k + (state_idx * HV + ihv) * W * K
    pd_tile = pending_d + meta * W * BV
    pc_tile = coeff + meta * W

    ptr_mixed = mixed_qkv + it * stride_mixed
    b_q = gl.load(ptr_mixed + ik * K + norm_k).to(gl.float32)
    b_k = gl.load(ptr_mixed + H * K + ik * K + norm_k).to(gl.float32)
    b_v = gl.load(ptr_mixed + 2 * H * K + ihv * V + ov).to(gl.float32)
    b_q = b_q / gl.sqrt(gl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / gl.sqrt(gl.sum(b_k * b_k) + 1e-6)
    b_q = gl.convert_layout(b_q * SCALE, gl.SliceLayout(0, state_layout))
    b_k = gl.convert_layout(b_k, gl.SliceLayout(0, state_layout))
    a_val = gl.load(a + it * stride_a + ihv).to(gl.float32)
    b_val = gl.load(b + it * stride_b + ihv).to(gl.float32)
    A_log_val = gl.load(A_log + ihv).to(gl.float32)
    dt_bias_val = gl.load(dt_bias + ihv).to(gl.float32)
    x = a_val + dt_bias_val
    softplus_x = gl.where(x <= 20.0, gl.log(1.0 + gl.exp(x)), x)
    g_val = -gl.exp(A_log_val) * softplus_x
    beta_val = 1.0 / (1.0 + gl.exp(-b_val))

    # Replay only live factors. A fixed W-trip loop wastes work at age zero.
    # Keep BF16 store/load rounding at every logical token boundary.
    for j in range(old_age):
        old_alpha = gl.load(pc_tile + j).to(gl.float32)
        old_k = gl.load(pk_head + j * K + ok).to(gl.float32)
        old_d = gl.load(pd_tile + j * BV + ov_local).to(gl.float32)
        # Stock consumes rounded H*alpha in its dot product before the
        # update. Pin the same FMA operand order for replay/materialize.
        scaled = b_h * old_alpha
        b_h = gl.fma(old_d[:, None], old_k[None, :], scaled)
        # Match the state store/load rounding at every stock token boundary.
        if base.dtype.element_ty == gl.bfloat16:
            b_h = b_h.to(gl.bfloat16).to(gl.float32)

    # Keep the stock packed FLA update/reduction source expressions and order.
    alpha = gl.exp(g_val)
    b_h *= alpha
    b_v -= gl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = gl.sum(b_h * b_q[None, :], 1)
    gl.store(output_ptr, b_o.to(out.dtype.element_ty))

    if FULL or old_age == W - 1:
        gl.store(ptr_base, b_h)
        gl.store(age + meta, 0)
        gl.store(prefix + meta, 1.0)
    else:
        if iv == 0:
            gl.store(pk_head + old_age * K + ok, b_k)
        gl.store(pd_tile + old_age * BV + ov_local, b_v)
        gl.store(pc_tile + old_age, alpha)
        gl.store(age + meta, old_age + 1)


def decode(mixed_qkv, a, b, A_log, dt_bias, base, out, indices, scratch,
           scale=128 ** -0.5, *, use_pdl=None, grouped_layout=None):
    """Launch after the caller has validated shapes, dtype and scratch."""
    heads = base.shape[1]
    rows = mixed_qkv.shape[0]
    # Small grids benefit from eight warps and immediate state writeback.
    # At higher concurrency W4 avoids enough stores to repay replay work.
    full = rows * heads <= 256
    if grouped_layout is None:
        grouped_layout = _USE_GROUPED
    if type(grouped_layout) is not bool:
        raise TypeError("grouped_layout must be bool or None")
    threshold = 768 if grouped_layout else 1024
    warps = 8 if full else (4 if rows * heads <= threshold else 2)
    state_rows = 4 if grouped_layout and not full else 1
    if use_pdl is None:
        use_pdl = _USE_PDL
    if type(use_pdl) is not bool:
        raise TypeError("use_pdl must be bool or None")
    return _ordered_decode[(4, heads, rows)](
        mixed_qkv, a, b, A_log, dt_bias, out, base, indices,
        scratch.pending_k, scratch.pending_d, scratch.coeff,
        scratch.prefix, scratch.age,
        mixed_qkv.stride(0), a.stride(0), b.stride(0), base.stride(0),
        indices.stride(0), base.shape[0], 4, scale, 16, heads, 128, 128,
        32, 4, full, state_rows, use_pdl, num_warps=warps, num_stages=3,
        launch_pdl=use_pdl)
