# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place presence penalties sourced from V2's persistent token counts."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv


@triton.jit
def _apply_presence_penalty_from_counts_kernel(
    logits_ptr,
    logits_stride,
    local_token_ids_ptr,
    local_token_ids_stride,
    output_counts_ptr,
    output_counts_stride,
    request_indices_ptr,
    presence_penalties_ptr,
    num_cols,
    counts_vocab_size,
    org_vocab_start,
    num_org_elements,
    num_org_elements_padded,
    added_vocab_start,
    num_added_elements,
    BLOCK_SIZE: tl.constexpr,
    INDEXED: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = cols < num_cols

    if INDEXED:
        local_ids = tl.load(
            local_token_ids_ptr + row * local_token_ids_stride + cols,
            mask=col_mask,
            other=0,
        )
    else:
        local_ids = cols

    is_org = local_ids < num_org_elements
    added_offsets = local_ids - num_org_elements_padded
    is_added = (added_offsets >= 0) & (added_offsets < num_added_elements)
    global_ids = tl.where(
        is_org,
        org_vocab_start + local_ids,
        added_vocab_start + added_offsets,
    )
    token_mask = col_mask & (is_org | is_added)
    token_mask &= (global_ids >= 0) & (global_ids < counts_vocab_size)

    request_idx = tl.load(request_indices_ptr + row)
    counts = tl.load(
        output_counts_ptr + request_idx * output_counts_stride + global_ids,
        mask=token_mask,
        other=0,
    )
    penalty = tl.load(presence_penalties_ptr + row)
    logits = tl.load(
        logits_ptr + row * logits_stride + cols,
        mask=col_mask,
        other=0.0,
    )
    logits -= tl.where(token_mask & (counts > 0), penalty, 0.0)
    tl.store(logits_ptr + row * logits_stride + cols, logits, mask=col_mask)


@triton.jit
def _apply_sparse_presence_penalty_kernel(
    logits_ptr,
    logits_stride,
    unique_token_ids_ptr,
    unique_token_ids_stride,
    num_unique_tokens_ptr,
    request_indices_ptr,
    presence_penalties_ptr,
    num_cols,
    org_vocab_start,
    num_org_elements,
    num_org_elements_padded,
    added_vocab_start,
    num_added_elements,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    request_idx = tl.load(request_indices_ptr + row)
    num_unique_tokens = tl.load(num_unique_tokens_ptr + request_idx)
    penalty = tl.load(presence_penalties_ptr + row)

    for start in tl.range(0, num_unique_tokens, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_unique_tokens
        global_ids = tl.load(
            unique_token_ids_ptr + request_idx * unique_token_ids_stride + offsets,
            mask=mask,
            other=0,
        )

        org_offsets = global_ids - org_vocab_start
        is_org = (org_offsets >= 0) & (org_offsets < num_org_elements)
        added_offsets = global_ids - added_vocab_start
        is_added = (added_offsets >= 0) & (added_offsets < num_added_elements)
        local_ids = tl.where(
            is_org,
            org_offsets,
            num_org_elements_padded + added_offsets,
        )
        token_mask = mask & (is_org | is_added)
        token_mask &= (local_ids >= 0) & (local_ids < num_cols)

        logits = tl.load(
            logits_ptr + row * logits_stride + local_ids,
            mask=token_mask,
            other=0.0,
        )
        tl.store(
            logits_ptr + row * logits_stride + local_ids,
            logits - penalty,
            mask=token_mask,
        )


def apply_presence_penalty_from_counts(
    logits: torch.Tensor,
    output_token_counts: torch.Tensor,
    request_indices: torch.Tensor,
    presence_penalties: torch.Tensor,
    *,
    org_vocab_start: int,
    num_org_elements: int,
    num_org_elements_padded: int,
    added_vocab_start: int,
    num_added_elements: int,
    local_token_ids: torch.Tensor | None = None,
) -> None:
    """Subtract a presence penalty without materializing a dense mask."""
    if logits.numel() == 0:
        return
    if logits.ndim != 2 or not logits.is_contiguous():
        raise ValueError("logits must be a contiguous rank-2 tensor")
    if output_token_counts.ndim != 2:
        raise ValueError("output_token_counts must be rank 2")
    if request_indices.shape != (logits.shape[0],):
        raise ValueError("request_indices must have one entry per logits row")
    if presence_penalties.shape != (logits.shape[0],):
        raise ValueError("presence_penalties must have one entry per logits row")
    if local_token_ids is not None and local_token_ids.shape != logits.shape:
        raise ValueError("local_token_ids must match logits")

    block_size = 256
    _apply_presence_penalty_from_counts_kernel[
        (logits.shape[0], cdiv(logits.shape[1], block_size))
    ](
        logits,
        logits.stride(0),
        local_token_ids,
        local_token_ids.stride(0) if local_token_ids is not None else 0,
        output_token_counts,
        output_token_counts.stride(0),
        request_indices,
        presence_penalties,
        logits.shape[1],
        output_token_counts.shape[1],
        org_vocab_start,
        num_org_elements,
        num_org_elements_padded,
        added_vocab_start,
        num_added_elements,
        BLOCK_SIZE=block_size,
        INDEXED=local_token_ids is not None,
    )


def apply_sparse_presence_penalty(
    logits: torch.Tensor,
    unique_token_ids: torch.Tensor,
    num_unique_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    presence_penalties: torch.Tensor,
    *,
    org_vocab_start: int,
    num_org_elements: int,
    num_org_elements_padded: int,
    added_vocab_start: int,
    num_added_elements: int,
) -> None:
    """Subtract presence penalties only at previously generated token ids."""
    if logits.numel() == 0:
        return
    if logits.ndim != 2 or not logits.is_contiguous():
        raise ValueError("logits must be a contiguous rank-2 tensor")
    if unique_token_ids.ndim != 2 or not unique_token_ids.is_contiguous():
        raise ValueError("unique_token_ids must be a contiguous rank-2 tensor")
    if num_unique_tokens.shape != (unique_token_ids.shape[0],):
        raise ValueError("num_unique_tokens must have one entry per request")
    if request_indices.shape != (logits.shape[0],):
        raise ValueError("request_indices must have one entry per logits row")
    if presence_penalties.shape != (logits.shape[0],):
        raise ValueError("presence_penalties must have one entry per logits row")

    _apply_sparse_presence_penalty_kernel[(logits.shape[0],)](
        logits,
        logits.stride(0),
        unique_token_ids,
        unique_token_ids.stride(0),
        num_unique_tokens,
        request_indices,
        presence_penalties,
        logits.shape[1],
        org_vocab_start,
        num_org_elements,
        num_org_elements_padded,
        added_vocab_start,
        num_added_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )
