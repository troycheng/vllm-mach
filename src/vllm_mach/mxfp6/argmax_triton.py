# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small Triton reductions for vocab-parallel greedy sampling."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


@triton.jit
def _block_argmax_kernel(
    LOGITS,
    BLOCK_VALUES,
    BLOCK_INDICES,
    VOCAB_START: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    ACTIVE_VOCAB_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    lane = tl.arange(0, BLOCK_SIZE)
    vocab_offsets = block_id * BLOCK_SIZE + lane
    mask = vocab_offsets < ACTIVE_VOCAB_SIZE

    values = tl.load(
        LOGITS + row * VOCAB_SIZE + vocab_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Padded dummy rows can contain NaNs; exclude them from the reduction.
    values = tl.where(values == values, values, -float("inf"))
    value, lane_idx = tl.max(values, axis=0, return_indices=True)

    token_offset = block_id * BLOCK_SIZE + lane_idx
    token_offset = tl.where(
        token_offset < ACTIVE_VOCAB_SIZE,
        token_offset,
        block_id * BLOCK_SIZE,
    )
    out_offset = row * NUM_BLOCKS + block_id
    tl.store(BLOCK_VALUES + out_offset, value)
    tl.store(BLOCK_INDICES + out_offset, VOCAB_START + token_offset)


@triton.jit
def _merge_argmax_kernel(
    BLOCK_VALUES,
    BLOCK_INDICES,
    OUT_VALUES,
    OUT_INDICES,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_NUM_BLOCKS)
    mask = lane < NUM_BLOCKS
    in_offset = row * NUM_BLOCKS + lane

    values = tl.load(BLOCK_VALUES + in_offset, mask=mask, other=-float("inf"))
    indices = tl.load(BLOCK_INDICES + in_offset, mask=mask, other=0)
    _, lane_idx = tl.max(values, axis=0, return_indices=True)
    token_id = tl.max(tl.where(lane == lane_idx, indices, 0), axis=0)

    tl.store(OUT_VALUES + row, tl.max(values, axis=0))
    tl.store(OUT_INDICES + row, token_id)


@triton.jit
def _global_pair_argmax_kernel(
    GATHERED_PAIRS,
    OUT_INDICES,
    PAIR_STRIDE_0: tl.constexpr,
    PAIR_STRIDE_1: tl.constexpr,
    TP_SIZE: tl.constexpr,
    BLOCK_TP: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_TP)
    mask = lane < TP_SIZE
    base = row * PAIR_STRIDE_0 + lane * 2 * PAIR_STRIDE_1

    values = tl.load(GATHERED_PAIRS + base, mask=mask, other=-float("inf"))
    indices = tl.load(GATHERED_PAIRS + base + PAIR_STRIDE_1, mask=mask, other=0)
    _, lane_idx = tl.max(values, axis=0, return_indices=True)
    token_id = tl.max(tl.where(lane == lane_idx, indices, 0.0), axis=0).to(tl.int32)
    tl.store(OUT_INDICES + row, token_id)


@triton.jit
def _indexed_argmax_kernel(
    VALUES,
    TOKEN_IDS,
    OUT_VALUES,
    OUT_TOKEN_IDS,
    VALUE_STRIDE_0: tl.constexpr,
    TOKEN_STRIDE_0: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
    INDEX_OFFSET: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_CANDIDATES)
    mask = lane < NUM_CANDIDATES
    values = tl.load(
        VALUES + row * VALUE_STRIDE_0 + lane,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    token_ids = tl.load(
        TOKEN_IDS + row * TOKEN_STRIDE_0 + lane,
        mask=mask,
        other=0x7FFFFFFF,
    ).to(tl.int32)
    values = tl.where(values == values, values, -float("inf"))
    max_value = tl.max(values, axis=0)
    min_token_id = tl.min(tl.where(values == max_value, token_ids, 0x7FFFFFFF), axis=0)
    tl.store(OUT_VALUES + row, max_value)
    tl.store(OUT_TOKEN_IDS + row, min_token_id + INDEX_OFFSET)


def local_argmax_triton(
    logits: torch.Tensor,
    *,
    vocab_start: int,
    active_vocab_size: int,
    block_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row local max values and global token ids."""
    assert logits.ndim == 2
    assert logits.is_cuda
    assert 0 < active_vocab_size <= logits.shape[1]
    if not logits.is_contiguous():
        logits = logits.contiguous()

    batch_size, vocab_size = logits.shape
    num_blocks = triton.cdiv(active_vocab_size, block_size)
    block_num_blocks = next_power_of_2(num_blocks)

    block_values = torch.empty(
        (batch_size, num_blocks), device=logits.device, dtype=torch.float32
    )
    block_indices = torch.empty(
        (batch_size, num_blocks), device=logits.device, dtype=torch.int32
    )
    out_values = torch.empty((batch_size,), device=logits.device, dtype=torch.float32)
    out_indices = torch.empty((batch_size,), device=logits.device, dtype=torch.int32)

    _block_argmax_kernel[(batch_size, num_blocks)](
        logits,
        block_values,
        block_indices,
        VOCAB_START=vocab_start,
        VOCAB_SIZE=vocab_size,
        ACTIVE_VOCAB_SIZE=active_vocab_size,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
    )
    _merge_argmax_kernel[(batch_size,)](
        block_values,
        block_indices,
        out_values,
        out_indices,
        NUM_BLOCKS=num_blocks,
        BLOCK_NUM_BLOCKS=block_num_blocks,
    )
    return out_values, out_indices


def indexed_argmax_triton(
    values: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    index_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce compact values with exact minimum-token-id tie breaking."""
    assert values.ndim == 2
    assert values.is_cuda
    assert values.shape == token_ids.shape
    assert 0 < values.shape[1] <= 1024
    if not values.is_contiguous():
        values = values.contiguous()
    if not token_ids.is_contiguous():
        token_ids = token_ids.contiguous()

    batch_size, num_candidates = values.shape
    out_values = torch.empty((batch_size,), device=values.device, dtype=torch.float32)
    out_token_ids = torch.empty((batch_size,), device=values.device, dtype=torch.int32)
    _indexed_argmax_kernel[(batch_size,)](
        values,
        token_ids,
        out_values,
        out_token_ids,
        VALUE_STRIDE_0=values.stride(0),
        TOKEN_STRIDE_0=token_ids.stride(0),
        NUM_CANDIDATES=num_candidates,
        BLOCK_CANDIDATES=next_power_of_2(num_candidates),
        INDEX_OFFSET=index_offset,
    )
    return out_values, out_token_ids


def reduce_global_argmax_triton(
    gathered_pairs: torch.Tensor,
    *,
    tp_size: int,
) -> torch.Tensor:
    """Reduce gathered ``(value, token_id)`` pairs to one token per row."""
    assert gathered_pairs.ndim == 2
    assert gathered_pairs.is_cuda
    assert gathered_pairs.shape[1] == tp_size * 2

    batch_size = gathered_pairs.shape[0]
    out_indices = torch.empty(
        (batch_size,), device=gathered_pairs.device, dtype=torch.int32
    )
    _global_pair_argmax_kernel[(batch_size,)](
        gathered_pairs,
        out_indices,
        gathered_pairs.stride(0),
        gathered_pairs.stride(1),
        TP_SIZE=tp_size,
        BLOCK_TP=next_power_of_2(tp_size),
    )
    return out_indices
