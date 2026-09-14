"""Fuse the second low-rank GEMM with the ordered dual-A4 output merge."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


M = 32
N = 17408
SUPPORTED_RANKS = (64,)
BLOCK_N = 64


@triton.jit
def _up_merge_kernel(
    terms_ptr,
    z_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, M)
    ranks = tl.arange(0, R)

    first = tl.load(terms_ptr + rows[:, None] * N + block_n[None, :]).to(tl.float32)
    second = tl.load(terms_ptr + (M + rows[:, None]) * N + block_n[None, :]).to(tl.float32)
    z = tl.load(z_ptr + rows[:, None] * R + ranks[None, :])
    b = tl.load(b_ptr + ranks[:, None] * N + block_n[None, :])
    low_rank = tl.dot(z, b, out_dtype=tl.float32)
    result = first + second * 0.25 + low_rank
    tl.store(out_ptr + rows[:, None] * N + block_n[None, :], result)


def _validate(name: str, value: torch.Tensor, shape: tuple[int, int], device: torch.device) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.bfloat16 or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be BF16 with shape {shape}")
    if not value.is_cuda or value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the shared CUDA device")


def up_merge(terms: torch.Tensor, z: torch.Tensor, b: torch.Tensor, block_n: int = BLOCK_N) -> torch.Tensor:
    """Return ``BF16((terms0 + terms1/4) + z @ b)`` for physical M32 only.

    The first low-rank GEMM that produces ``z`` remains outside this kernel.
    ``block_n`` is deliberately fixed at 64: it is part of the measured kernel
    contract, not an autotuned launch choice.
    """
    if block_n != BLOCK_N:
        raise ValueError(f"only fixed block_n={BLOCK_N} is supported")
    if not isinstance(z, torch.Tensor) or z.ndim != 2:
        raise TypeError("z must be a rank-2 tensor")
    rank = int(z.shape[1])
    if rank not in SUPPORTED_RANKS:
        raise ValueError(f"R must be one of {SUPPORTED_RANKS}, got {rank}")
    device = z.device
    _validate("terms", terms, (2 * M, N), device)
    _validate("z", z, (M, rank), device)
    _validate("b", b, (rank, N), device)

    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    _up_merge_kernel[(triton.cdiv(N, BLOCK_N),)](
        terms,
        z,
        b,
        output,
        M=M,
        N=N,
        R=rank,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
