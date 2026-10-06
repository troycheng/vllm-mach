# SPDX-License-Identifier: Apache-2.0
"""Ordered raw-alpha FP32 GDN kernels for the frozen TP1 SM120 profile.

Caller-owned W4 scratch; arithmetic and launch configurations are fixed.
"""
import torch
from vllm.triton_utils import tl, triton
from vllm.third_party.flash_linear_attention.ops.op import exp
from . import ordered_allm_triton as ordered

HV,H,V,K,BV,NV,QKV,W=32,16,128,128,32,4,8192,4
DeferredScratch=ordered.DeferredScratch
required_shapes=ordered.required_shapes
validate=ordered.validate
reset_slots=ordered.reset_slots
materialize_slots=ordered.materialize_slots
_LAST_KERNEL=None
_COMPILED={}

@triton.jit
def _lowm_ordered_decode(mixed_qkv, a, b, A_log, dt_bias, out, base, indices,
                    pending_k, pending_d, coeff, prefix, age,
                    stride_mixed: tl.constexpr, stride_a: tl.constexpr,
                    stride_b: tl.constexpr, stride_base_page: tl.constexpr,
                    stride_index: tl.constexpr, P: tl.constexpr, W: tl.constexpr,
                    SCALE: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                    V: tl.constexpr, K: tl.constexpr, BV: tl.constexpr,
                    NV: tl.constexpr):
    iv = tl.program_id(0)
    nh = tl.program_id(1)
    it, ihv = nh // HV, nh % HV
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
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
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
    alpha = exp(g_val)
    b_h *= alpha
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(output_ptr, b_o.to(out.dtype.element_ty))

    # Low-M fallback never defers: current output and full dense state are
    # committed in stock order, with independent metadata for every iv tile.
    tl.store(ptr_base, b_h)
    tl.store(age + meta, 0)
    tl.store(prefix + meta, 1.0)


def decode(mixed_qkv,a,b,A_log,dt_bias,base,out,indices,scratch,scale=K**-0.5):
    p,w=validate(base,scratch)
    ordered._check_indices(indices,base)
    m=int(mixed_qkv.shape[0])
    if m not in (1,2,4,8,16) or tuple(mixed_qkv.shape)!=(m,QKV):
        raise ValueError("only packed M1/2/4/8/16 QKV8192")
    for name,t,shape,dtype in (("qkv",mixed_qkv,(m,QKV),torch.bfloat16),
                               ("a",a,(m,HV),torch.bfloat16),
                               ("b",b,(m,HV),torch.bfloat16)):
        if tuple(t.shape)!=shape or t.dtype!=dtype or t.device!=base.device or t.stride(1)!=1 or t.stride(0)<=0:
            raise ValueError(f"invalid {name} BF16 layout/device")
    for name,t,dtype in (("A_log",A_log,torch.float32),("dt_bias",dt_bias,torch.bfloat16)):
        if tuple(t.shape)!=(HV,) or t.dtype!=dtype or t.device!=base.device or t.stride(0)!=1:
            raise ValueError(f"invalid {name}")
    if tuple(out.shape)!=(m,1,HV,V) or out.dtype!=torch.bfloat16 or out.device!=base.device or not out.is_contiguous() or tuple(indices.shape)!=(m,):
        raise ValueError("out contiguous BF16 [M,1,32,128], indices[M] required")
    if not ordered.math_isfinite_positive(scale):raise ValueError("scale must be positive finite")
    global _LAST_KERNEL
    _LAST_KERNEL=_lowm_ordered_decode[(NV,HV*m)](
        mixed_qkv,a,b,A_log,dt_bias,out,base,indices,
        scratch.pending_k,scratch.pending_d,scratch.coeff,scratch.prefix,scratch.age,
        mixed_qkv.stride(0),a.stride(0),b.stride(0),base.stride(0),indices.stride(0),
        p,w,scale,H,HV,V,K,BV,NV,num_warps=1,num_stages=3)
    _COMPILED[str(getattr(_LAST_KERNEL,"hash",len(_COMPILED)))]=_LAST_KERNEL
    return out
