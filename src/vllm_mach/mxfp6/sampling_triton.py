# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact top-k/top-p and shard-local Gumbel sampling, ported from vllm-shpgy."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


@triton.jit
def _rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    u = tl.minimum(u, 0.9999999999999999)
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def _full_vocab_sample_block_kernel(
    LOGITS,
    SEEDS,
    EXCLUDE_TOKEN_IDS,
    SHARD_TOKEN_IDS,
    BLOCK_VALUES,
    BLOCK_INDICES,
    VOCAB_START: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    ACTIVE_VOCAB_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SCALE_OVER_TEMPERATURE: tl.constexpr,
    HAS_EXCLUDE: tl.constexpr,
    HAS_SHARD_TOKEN_IDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    lane = tl.arange(0, BLOCK_SIZE)
    vocab_offsets = block_id * BLOCK_SIZE + lane
    mask = vocab_offsets < ACTIVE_VOCAB_SIZE
    if HAS_SHARD_TOKEN_IDS:
        global_offsets = tl.load(
            SHARD_TOKEN_IDS + vocab_offsets,
            mask=vocab_offsets < VOCAB_SIZE,
            other=-1,
        )
        mask = mask & (global_offsets >= 0)
    else:
        global_offsets = VOCAB_START + vocab_offsets
    if HAS_EXCLUDE:
        exclude_token_id = tl.load(EXCLUDE_TOKEN_IDS + row)
        mask = mask & (global_offsets != exclude_token_id)

    logits = tl.load(
        LOGITS + row * VOCAB_SIZE + vocab_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    logits = logits * SCALE_OVER_TEMPERATURE

    seed = tl.load(SEEDS + row)
    random_offsets = tl.where(mask, global_offsets, 0)
    u = _rand64(seed, random_offsets, includes_zero=False)
    gumbel_noise = -tl.log(-tl.log(u))
    scores = logits.to(tl.float64) + gumbel_noise
    scores = tl.where(mask, scores, -float("inf"))

    value, lane_idx = tl.max(scores, axis=0, return_indices=True)
    token_id = tl.max(tl.where((lane == lane_idx) & mask, global_offsets, 0), axis=0)
    out_offset = row * NUM_BLOCKS + block_id
    tl.store(BLOCK_VALUES + out_offset, value)
    tl.store(BLOCK_INDICES + out_offset, token_id)


@triton.jit
def _merge_full_vocab_sample_blocks_kernel(
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


def sample_full_vocab_from_shard_triton(
    logits: torch.Tensor,
    *,
    vocab_start: int,
    active_vocab_size: int,
    seeds: torch.Tensor,
    shard_token_ids: torch.Tensor | None = None,
    exclude_token_ids: torch.Tensor | None = None,
    scale: float = 1.0,
    temperature: float = 1.0,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample local vocab-shard winners with Gumbel-max.

    Returns per-row ``(score, global_token_id)`` winners. Reducing those winners
    across TP ranks is equivalent to sampling from the full global softmax,
    while communicating only one pair per row per rank.
    """
    assert logits.ndim == 2
    assert logits.is_cuda
    assert seeds.ndim == 1 and seeds.shape[0] == logits.shape[0]
    assert seeds.dtype == torch.int64
    if exclude_token_ids is not None:
        assert exclude_token_ids.ndim == 1
        assert exclude_token_ids.shape[0] == logits.shape[0]
        assert exclude_token_ids.dtype in (torch.int32, torch.int64)
    assert 0 < active_vocab_size <= logits.shape[1]
    assert temperature > 0.0
    if shard_token_ids is not None:
        assert shard_token_ids.ndim == 1
        assert shard_token_ids.shape[0] == logits.shape[1]
        assert shard_token_ids.dtype == torch.int64

    batch_size, vocab_size = logits.shape
    out_values = torch.empty((batch_size,), device=logits.device, dtype=torch.float64)
    out_indices = torch.empty((batch_size,), device=logits.device, dtype=torch.int32)
    if batch_size == 0:
        return out_values, out_indices

    if not logits.is_contiguous():
        logits = logits.contiguous()
    if exclude_token_ids is None:
        exclude_token_ids = seeds
        has_exclude = False
    else:
        if not exclude_token_ids.is_contiguous():
            exclude_token_ids = exclude_token_ids.contiguous()
        has_exclude = True
    if shard_token_ids is None:
        shard_token_ids = seeds
        has_shard_token_ids = False
    else:
        if not shard_token_ids.is_contiguous():
            shard_token_ids = shard_token_ids.contiguous()
        has_shard_token_ids = True

    num_blocks = triton.cdiv(active_vocab_size, block_size)
    block_values = torch.empty(
        (batch_size, num_blocks), device=logits.device, dtype=torch.float64
    )
    block_indices = torch.empty(
        (batch_size, num_blocks), device=logits.device, dtype=torch.int32
    )
    _full_vocab_sample_block_kernel[(batch_size, num_blocks)](
        logits,
        seeds,
        exclude_token_ids,
        shard_token_ids,
        block_values,
        block_indices,
        VOCAB_START=int(vocab_start),
        VOCAB_SIZE=vocab_size,
        ACTIVE_VOCAB_SIZE=active_vocab_size,
        NUM_BLOCKS=num_blocks,
        SCALE_OVER_TEMPERATURE=float(scale) / float(temperature),
        HAS_EXCLUDE=has_exclude,
        HAS_SHARD_TOKEN_IDS=has_shard_token_ids,
        BLOCK_SIZE=block_size,
    )
    _merge_full_vocab_sample_blocks_kernel[(batch_size,)](
        block_values,
        block_indices,
        out_values,
        out_indices,
        NUM_BLOCKS=num_blocks,
        BLOCK_NUM_BLOCKS=next_power_of_2(num_blocks),
    )
    return out_values, out_indices


@triton.jit
def _pack_topk_pairs_kernel(
    LOCAL_VALS,
    LOCAL_INDICES,
    LOCAL_PAIRS,
    VOCAB_START: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < TOP_K

    input_base = row_id * TOP_K
    vals = tl.load(LOCAL_VALS + input_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    indices = tl.load(LOCAL_INDICES + input_base + offsets, mask=mask, other=0).to(
        tl.int64
    )
    global_indices = indices + VOCAB_START

    output_base = row_id * TOP_K * 2
    tl.store(LOCAL_PAIRS + output_base + offsets * 2, vals, mask=mask)
    tl.store(
        LOCAL_PAIRS + output_base + offsets * 2 + 1,
        global_indices.to(tl.float32),
        mask=mask,
    )


def pack_topk_pairs_triton(
    local_vals: torch.Tensor,
    local_indices: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    """Pack local top-k values and local ids into compact gather pairs.

    The output shape is ``[batch, top_k * 2]`` with interleaved
    ``(logit, global_token_id_as_float)`` entries, matching the distributed
    gather send buffer consumed by compact top-k sampling.
    """
    assert local_vals.ndim == 2
    assert local_indices.shape == local_vals.shape
    assert local_vals.dtype == torch.float32
    assert local_indices.dtype == torch.int64

    batch_size, top_k = local_vals.shape
    local_pairs = torch.empty(
        (batch_size, top_k * 2), dtype=torch.float32, device=local_vals.device
    )
    if batch_size == 0 or top_k == 0:
        return local_pairs

    if not local_vals.is_contiguous():
        local_vals = local_vals.contiguous()
    if not local_indices.is_contiguous():
        local_indices = local_indices.contiguous()

    block = next_power_of_2(top_k)
    _pack_topk_pairs_kernel[(batch_size,)](
        local_vals,
        local_indices,
        local_pairs,
        VOCAB_START=int(vocab_start),
        TOP_K=top_k,
        BLOCK=block,
    )
    return local_pairs


@triton.jit
def _select_from_compact_topk_pairs_kernel(
    GATHERED_PAIRS,
    TOP_VALS_OUT,
    TOP_IDS_OUT,
    NUM_CANDIDATES: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_P: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cand_offsets = tl.arange(0, CANDIDATE_BLOCK)
    cand_mask = cand_offsets < NUM_CANDIDATES
    pair_base = row_id * NUM_CANDIDATES * 2
    vals = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2,
        mask=cand_mask,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2 + 1,
        mask=cand_mask,
        other=0.0,
    ).to(tl.float32)

    top_offsets = tl.arange(0, TOPK_BLOCK)
    top_vals = tl.full((TOPK_BLOCK,), -float("inf"), tl.float32)
    top_ids = tl.full((TOPK_BLOCK,), 0.0, tl.float32)
    work_vals = vals

    for i in tl.static_range(0, TOP_K):
        max_val, max_idx = tl.max(work_vals, axis=0, return_indices=True)
        is_max = cand_offsets == max_idx
        token_id = tl.sum(tl.where(is_max, ids, 0.0), axis=0)
        top_vals = tl.where(top_offsets == i, max_val, top_vals)
        top_ids = tl.where(top_offsets == i, token_id, top_ids)
        work_vals = tl.where(is_max, -float("inf"), work_vals)

    valid_top = top_offsets < TOP_K
    if TOP_P < 1.0:
        max_top_val = tl.max(tl.where(valid_top, top_vals, -float("inf")), axis=0)
        weights = tl.exp(top_vals - max_top_val)
        weights = tl.where(valid_top, weights, 0.0)
        denom = tl.sum(weights, axis=0)
        probs = tl.where(denom > 0.0, weights / denom, 0.0)
        prev_cum_probs = tl.cumsum(probs, axis=0) - probs
        valid_top = valid_top & (prev_cum_probs <= TOP_P)

    out_base = row_id * TOP_K + top_offsets
    out_mask = top_offsets < TOP_K
    top_vals = tl.where(valid_top, top_vals, -float("inf"))
    tl.store(TOP_VALS_OUT + out_base, top_vals, mask=out_mask)
    tl.store(TOP_IDS_OUT + out_base, top_ids.to(tl.int64), mask=out_mask)


def select_compact_topk_pairs_triton(
    gathered_pairs: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse global compact top-k merge and top-p filtering.

    ``gathered_pairs`` is the TP-gathered compact candidate tensor with shape
    ``[batch, num_candidates, 2]`` where the last dimension stores
    ``(logit, token_id_as_float)``. The outputs have shape ``[batch, top_k]``.
    """
    assert gathered_pairs.ndim == 3
    assert gathered_pairs.shape[-1] == 2
    assert gathered_pairs.dtype == torch.float32
    assert 0 < top_k <= 64

    batch_size, num_candidates, _ = gathered_pairs.shape
    top_vals = torch.empty(
        (batch_size, top_k), dtype=torch.float32, device=gathered_pairs.device
    )
    top_ids = torch.empty(
        (batch_size, top_k), dtype=torch.int64, device=gathered_pairs.device
    )
    if batch_size == 0:
        return top_vals, top_ids
    if num_candidates < top_k:
        raise ValueError(
            f"num_candidates ({num_candidates}) must be >= top_k ({top_k})"
        )
    if num_candidates > 2048:
        raise ValueError(
            "compact top-k fused select only supports up to 2048 "
            f"candidates, got {num_candidates}"
        )

    if not gathered_pairs.is_contiguous():
        gathered_pairs = gathered_pairs.contiguous()

    candidate_block = next_power_of_2(num_candidates)
    topk_block = next_power_of_2(top_k)
    _select_from_compact_topk_pairs_kernel[(batch_size,)](
        gathered_pairs,
        top_vals,
        top_ids,
        NUM_CANDIDATES=num_candidates,
        TOP_K=top_k,
        TOP_P=float(top_p),
        CANDIDATE_BLOCK=candidate_block,
        TOPK_BLOCK=topk_block,
    )
    return top_vals, top_ids


@triton.jit
def _sample_from_compact_topk_pairs_kernel(
    GATHERED_PAIRS,
    SEEDS,
    SAMPLED,
    NUM_CANDIDATES: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_P: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cand_offsets = tl.arange(0, CANDIDATE_BLOCK)
    cand_mask = cand_offsets < NUM_CANDIDATES
    pair_base = row_id * NUM_CANDIDATES * 2
    vals = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2,
        mask=cand_mask,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2 + 1,
        mask=cand_mask,
        other=0.0,
    ).to(tl.float32)

    top_offsets = tl.arange(0, TOPK_BLOCK)
    top_vals = tl.full((TOPK_BLOCK,), -float("inf"), tl.float32)
    top_ids = tl.full((TOPK_BLOCK,), 0.0, tl.float32)
    work_vals = vals

    for i in tl.static_range(0, TOP_K):
        max_val, max_idx = tl.max(work_vals, axis=0, return_indices=True)
        is_max = cand_offsets == max_idx
        token_id = tl.sum(tl.where(is_max, ids, 0.0), axis=0)
        top_vals = tl.where(top_offsets == i, max_val, top_vals)
        top_ids = tl.where(top_offsets == i, token_id, top_ids)
        work_vals = tl.where(is_max, -float("inf"), work_vals)

    valid_top = top_offsets < TOP_K
    if TOP_P >= 1.0:
        keep = valid_top
    else:
        max_top_val = tl.max(tl.where(valid_top, top_vals, -float("inf")), axis=0)
        weights = tl.exp(top_vals - max_top_val)
        weights = tl.where(valid_top, weights, 0.0)
        denom = tl.sum(weights, axis=0)
        probs = tl.where(denom > 0.0, weights / denom, 0.0)
        prev_cum_probs = tl.cumsum(probs, axis=0) - probs
        keep = valid_top & (prev_cum_probs <= TOP_P)

    seed = tl.load(SEEDS + row_id)
    u = _rand64(seed, top_offsets, includes_zero=False)
    gumbel_noise = -tl.log(-tl.log(u))
    scores = tl.where(keep, top_vals.to(tl.float64) + gumbel_noise, -float("inf"))
    _, sampled_idx = tl.max(scores, axis=0, return_indices=True)
    sampled_id = tl.sum(tl.where(top_offsets == sampled_idx, top_ids, 0.0), axis=0)
    tl.store(SAMPLED + row_id, sampled_id.to(tl.int64))


def sample_from_compact_topk_pairs_triton(
    gathered_pairs: torch.Tensor,
    top_k: int,
    top_p: float,
    seeds: torch.Tensor,
) -> torch.Tensor:
    """Fuse global compact top-k merge, top-p filtering, and sampling.

    ``gathered_pairs`` is the TP-gathered compact candidate tensor with shape
    ``[batch, num_candidates, 2]`` where the last dimension stores
    ``(logit, token_id_as_float)``. The output is one sampled global token id
    per row.
    """
    assert gathered_pairs.ndim == 3
    assert gathered_pairs.shape[-1] == 2
    assert gathered_pairs.dtype == torch.float32
    assert seeds.ndim == 1 and seeds.shape[0] == gathered_pairs.shape[0]
    assert seeds.dtype == torch.int64
    assert 0 < top_k <= 64

    batch_size, num_candidates, _ = gathered_pairs.shape
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int64, device=gathered_pairs.device)
    if num_candidates < top_k:
        raise ValueError(
            f"num_candidates ({num_candidates}) must be >= top_k ({top_k})"
        )
    if num_candidates > 2048:
        raise ValueError(
            "compact top-k fused sampling only supports up to 2048 "
            f"candidates, got {num_candidates}"
        )

    if not gathered_pairs.is_contiguous():
        gathered_pairs = gathered_pairs.contiguous()

    sampled = torch.empty(
        (batch_size,), dtype=torch.int64, device=gathered_pairs.device
    )
    candidate_block = next_power_of_2(num_candidates)
    topk_block = next_power_of_2(top_k)
    _sample_from_compact_topk_pairs_kernel[(batch_size,)](
        gathered_pairs,
        seeds,
        sampled,
        NUM_CANDIDATES=num_candidates,
        TOP_K=top_k,
        TOP_P=float(top_p),
        CANDIDATE_BLOCK=candidate_block,
        TOPK_BLOCK=topk_block,
    )
    return sampled
