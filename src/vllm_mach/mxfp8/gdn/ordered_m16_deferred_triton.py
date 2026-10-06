# SPDX-License-Identifier: Apache-2.0
"""Ordered raw-alpha FP32 GDN kernels for the frozen TP1 SM120 profile.

Caller-owned W4 scratch; arithmetic and launch configurations are fixed.
"""
from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton
from vllm.third_party.flash_linear_attention.ops.op import exp

HV, H, V, K, BV = 32, 16, 128, 128, 32
NV = V // BV
QKV = (2 * H * K) + HV * V
W = 4
_LAST_KERNEL = None
_LAST_MATERIALIZE_KERNEL = None


@dataclass
class DeferredScratch:
    pending_k: torch.Tensor  # [P,HV,W,K] FP32; Vtile0 owns current write
    pending_d: torch.Tensor  # [P,HV,NV,W,BV] FP32
    coeff: torch.Tensor      # [P,HV,NV,W] FP32 raw exp(g), not cumulative
    prefix: torch.Tensor     # [P,HV,NV] FP32 compatibility storage, unused
    age: torch.Tensor        # [P,HV,NV] int32


def required_shapes(pages: int, window: int) -> dict:
    if pages < 2 or window != W:
        raise ValueError("ordered component supports >=2 pages and W4 only")
    return {"pending_k": (pages, HV, W, K),
            "pending_d": (pages, HV, NV, W, BV),
            "coeff": (pages, HV, NV, W),
            "prefix": (pages, HV, NV), "age": (pages, HV, NV)}


def validate(base: torch.Tensor, scratch: DeferredScratch) -> tuple[int, int]:
    if base.ndim != 4 or tuple(base.shape[1:]) != (HV, V, K):
        raise ValueError("base must be [P,32,128,128]")
    if base.dtype != torch.float32 or not base.is_cuda:
        raise ValueError("base must be CUDA FP32")
    if base.stride(3) != 1 or base.stride(2) != K or base.stride(1) != V*K:
        raise ValueError("base inner head,V,K dimensions must be dense")
    if base.stride(0) < HV*V*K:
        raise ValueError("base page stride overlaps another slot")
    p = int(base.shape[0])
    specs = required_shapes(p, int(scratch.pending_k.shape[2]))
    for name, shape in specs.items():
        tensor = getattr(scratch, name)
        dtype = torch.int32 if name == "age" else torch.float32
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                or not tensor.is_cuda or tensor.device != base.device
                or not tensor.is_contiguous()):
            raise ValueError(f"invalid caller-owned {name}: {shape} {dtype}")
    return p, W


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


@triton.jit
def _ordered_materialize(base, indices, pending_k, pending_d, coeff, prefix,
                         age, stride_base_page: tl.constexpr,
                         stride_index: tl.constexpr, P: tl.constexpr,
                         W: tl.constexpr, HV: tl.constexpr, V: tl.constexpr,
                         K: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr):
    iv, ihv, it = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    state_idx = tl.load(indices + it * stride_index).to(tl.int32)
    if (state_idx <= 0) | (state_idx >= P):
        return
    meta = (state_idx * HV + ihv) * NV + iv
    old_age = tl.load(age + meta)
    if (old_age <= 0) | (old_age >= W):
        return
    ok, ov_local = tl.arange(0, K), tl.arange(0, BV)
    ov = iv * BV + ov_local
    ptr_base = base + state_idx * stride_base_page + ihv * V * K + ov[:, None] * K + ok[None, :]
    b_h = tl.load(ptr_base).to(tl.float32)
    pk_head = pending_k + (state_idx * HV + ihv) * W * K
    pd_tile = pending_d + meta * W * BV
    pc_tile = coeff + meta * W
    for j in range(W):
        if j < old_age:
            old_alpha = tl.load(pc_tile + j).to(tl.float32)
            old_k = tl.load(pk_head + j * K + ok).to(tl.float32)
            old_d = tl.load(pd_tile + j * BV + ov_local).to(tl.float32)
            scaled = b_h * old_alpha
            b_h = tl.fma(old_d[:, None], old_k[None, :], scaled)
    tl.store(ptr_base, b_h)
    tl.store(prefix + meta, 1.0)
    tl.store(age + meta, 0)


@triton.jit
def _ordered_reset(base, indices, prefix, age,
                   stride_base_page: tl.constexpr,
                   stride_index: tl.constexpr, P: tl.constexpr,
                   ZERO_BASE: tl.constexpr, HV: tl.constexpr,
                   V: tl.constexpr, K: tl.constexpr, BV: tl.constexpr,
                   NV: tl.constexpr):
    iv, ihv, it = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    state_idx = tl.load(indices + it * stride_index).to(tl.int32)
    if (state_idx <= 0) | (state_idx >= P):
        return
    meta = (state_idx * HV + ihv) * NV + iv
    tl.store(age + meta, 0)
    tl.store(prefix + meta, 1.0)
    if ZERO_BASE:
        ok, ov = tl.arange(0, K), iv * BV + tl.arange(0, BV)
        ptr_base = base + state_idx * stride_base_page + ihv * V * K + ov[:, None] * K + ok[None, :]
        tl.store(ptr_base, tl.full((BV, K), 0, tl.float32))


def _check_indices(indices: torch.Tensor, base: torch.Tensor):
    if (indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64)
            or not indices.is_cuda or indices.device != base.device
            or indices.stride(0) <= 0):
        raise ValueError("indices must be CUDA int32/int64 [M] with positive stride")
    # Caller guarantees unique positive slots; zero is the invalid sentinel.


def decode(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
           A_log: torch.Tensor, dt_bias: torch.Tensor, base: torch.Tensor,
           out: torch.Tensor, indices: torch.Tensor, scratch: DeferredScratch,
           scale: float = K ** -0.5):
    p, w = validate(base, scratch)
    _check_indices(indices, base)
    m = int(mixed_qkv.shape[0])
    if m != 16 or tuple(mixed_qkv.shape) != (m, QKV):
        raise ValueError("only packed M16 QKV8192")
    if (mixed_qkv.dtype != torch.bfloat16 or not mixed_qkv.is_cuda
            or mixed_qkv.device != base.device or mixed_qkv.stride(1) != 1):
        raise ValueError("mixed_qkv must be CUDA BF16 [M,8192] stride1")
    if (tuple(a.shape) != (m, HV) or tuple(b.shape) != (m, HV)
            or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
            or a.device != base.device or b.device != base.device
            or a.stride(1) != 1 or b.stride(1) != 1):
        raise ValueError("a/b must be CUDA BF16 [M,32] stride1")
    if (tuple(A_log.shape) != (HV,) or A_log.dtype != torch.float32
            or tuple(dt_bias.shape) != (HV,) or dt_bias.dtype != torch.bfloat16
            or A_log.device != base.device or dt_bias.device != base.device
            or A_log.stride(0) != 1 or dt_bias.stride(0) != 1):
        raise ValueError("A_log FP32 and dt_bias BF16 need 32 contiguous heads")
    if (tuple(out.shape) != (m, 1, HV, V) or out.dtype != torch.bfloat16
            or out.device != base.device or not out.is_contiguous()
            or tuple(indices.shape) != (m,)):
        raise ValueError("out contiguous BF16 [M,1,32,128], indices [M]")
    if not math_isfinite_positive(scale):
        raise ValueError("scale must be finite and positive")
    global _LAST_KERNEL
    _LAST_KERNEL = _ordered_decode[(NV, HV, m)](
        mixed_qkv, a, b, A_log, dt_bias, out, base, indices,
        scratch.pending_k, scratch.pending_d, scratch.coeff,
        scratch.prefix, scratch.age,
        mixed_qkv.stride(0), a.stride(0), b.stride(0), base.stride(0),
        indices.stride(0), p, w, scale, H, HV, V, K, BV, NV,
        num_warps=1, num_stages=3)
    return out


def math_isfinite_positive(value):
    import math
    return math.isfinite(value) and value > 0


def materialize_slots(base: torch.Tensor, indices: torch.Tensor,
                      scratch: DeferredScratch):
    """Apply raw pending factors on this stream before stock reads base."""
    p, w = validate(base, scratch)
    _check_indices(indices, base)
    global _LAST_MATERIALIZE_KERNEL
    _LAST_MATERIALIZE_KERNEL = _ordered_materialize[(NV, HV, indices.numel())](
        base, indices, scratch.pending_k, scratch.pending_d, scratch.coeff,
        scratch.prefix, scratch.age, base.stride(0), indices.stride(0), p, w,
        HV, V, K, BV, NV, num_warps=1, num_stages=3)


def reset_slots(base: torch.Tensor, indices: torch.Tensor,
                scratch: DeferredScratch, *, zero_base: bool = False):
    """Discard pending terms on assignment/recycle; optionally clear base."""
    p, _ = validate(base, scratch)
    _check_indices(indices, base)
    _ordered_reset[(NV, HV, indices.numel())](
        base, indices, scratch.prefix, scratch.age,
        base.stride(0), indices.stride(0), p, zero_base, HV, V, K, BV, NV,
        num_warps=1, num_stages=3)
