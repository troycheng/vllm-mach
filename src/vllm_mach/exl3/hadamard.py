# SPDX-License-Identifier: Apache-2.0

"""Memory-bounded EXL3 Hadamard weight folding.

This is the only reconstruction helper needed by the dense provider.  Keeping
it here avoids a runtime dependency on the experimental FP4 conversion overlay.
"""

from __future__ import annotations

import math
import os

import torch

_HADAMARD_BLOCK = 128
_HADAMARD_NORM = 1.0 / math.sqrt(_HADAMARD_BLOCK)
_HADAMARD_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _hadamard_128_matrix(
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    key = (str(device), str(dtype))
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < _HADAMARD_BLOCK:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    matrix = matrix * _HADAMARD_NORM
    if dtype != torch.float32:
        matrix = matrix.to(dtype)
    _HADAMARD_CACHE[key] = matrix
    return matrix


def hadamard_fold_weight_chunked(
    weight: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    """Fold EXL3 input/output Hadamards and scales into ``weight`` in place."""

    k, n = weight.shape
    if k % _HADAMARD_BLOCK or n % _HADAMARD_BLOCK:
        raise ValueError(
            f"K and N must be multiples of {_HADAMARD_BLOCK}, got K={k}, N={n}"
        )
    k_blocks = k // _HADAMARD_BLOCK
    n_blocks = n // _HADAMARD_BLOCK

    if suh.numel() == k:
        suh_elements = suh
    elif suh.numel() == k_blocks:
        suh_elements = suh.repeat_interleave(_HADAMARD_BLOCK)
    else:
        raise ValueError(f"suh length {suh.numel()} != K={k} or K//128={k_blocks}")
    if svh.numel() == n:
        svh_elements = svh
    elif svh.numel() == n_blocks:
        svh_elements = svh.repeat_interleave(_HADAMARD_BLOCK)
    else:
        raise ValueError(f"svh length {svh.numel()} != N={n} or N//128={n_blocks}")

    hadamard = _hadamard_128_matrix(weight.device)
    suh_f32 = suh_elements.float().reshape(k, 1)
    svh_f32 = svh_elements.float().reshape(1, n)
    budget_mb = int(os.environ.get("VLLM_EXL3_FOLD_FP32_BUDGET_MB", "96"))
    rows_per_chunk = max(
        _HADAMARD_BLOCK,
        (budget_mb * 1024 * 1024) // max(1, n * 4 * 3),
    )
    blocks_per_chunk = max(
        1,
        min(k_blocks, rows_per_chunk // _HADAMARD_BLOCK),
    )
    for block_start in range(0, k_blocks, blocks_per_chunk):
        block_count = min(blocks_per_chunk, k_blocks - block_start)
        row_start = block_start * _HADAMARD_BLOCK
        row_end = row_start + block_count * _HADAMARD_BLOCK
        chunk = weight[row_start:row_end].float()
        blocked = chunk.reshape(
            block_count,
            _HADAMARD_BLOCK,
            n_blocks,
            _HADAMARD_BLOCK,
        )
        left = torch.einsum("ab,ibjd->iajd", hadamard, blocked)
        folded = torch.einsum("iajb,bc->iajc", left, hadamard)
        folded = (
            folded.reshape(block_count * _HADAMARD_BLOCK, n)
            * suh_f32[row_start:row_end]
            * svh_f32
        )
        weight[row_start:row_end] = folded.to(weight.dtype)
    return weight
