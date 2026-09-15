# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared candidate selection and BF16 refinement for hybrid LM heads."""

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer

logger = init_logger(__name__)
_TOPK_CANDIDATE_MULTIPLIER = 8
_MAX_AUTO_CANDIDATES = 1024
_DEFAULT_AUTOTUNE_MAX_ROWS = 2048


def _candidate_count_for_topk(
    configured_candidates: int,
    top_k: int,
    *,
    output_size: int,
) -> int:
    """Return a cheap, accuracy-oriented candidate width for ``top_k``.

    The persistent MXFP weight is independent of the candidate width, so a
    top-k request can afford to refine more coarse candidates without another
    quantized copy.  Keep the configured width for greedy/MTP and the common
    small top-k path; for larger top-k requests expand in power-of-two steps.
    The 1024 cap matches the fast indexed reduction kernels.  Requests above
    that width are rejected by ``can_use`` and use the original full-logit
    path instead.
    """
    if top_k <= 0 or configured_candidates >= _MAX_AUTO_CANDIDATES:
        return min(configured_candidates, output_size)

    # Do not penalize the default top-k=20 path when the configured width is
    # 128.  Expansion starts once the requested top-k is large enough that a
    # wider candidate set materially improves recall (e.g. top-k=40).
    desired = max(configured_candidates, top_k * _TOPK_CANDIDATE_MULTIPLIER)
    if desired <= configured_candidates * 2:
        return min(configured_candidates, output_size)

    if desired >= _MAX_AUTO_CANDIDATES:
        expanded = _MAX_AUTO_CANDIDATES
    else:
        expanded = 1 << (desired - 1).bit_length()
    return min(max(configured_candidates, expanded), output_size)


def _select_indexed_bf16_candidate_tile(
    num_rows: int,
    num_candidates: int,
    input_size: int,
) -> int:
    # The tiled kernel keeps the same FP32 reduction order for each candidate;
    # it only shares the hidden load across a small group of candidates.  The
    # old large-hidden guard forced the 27B (K=5120) lm-head through one
    # program per candidate, even though small candidate tiles are consistently
    # faster for mixed/decode row counts and produce bit-identical BF16 results.
    if num_rows < 8 or num_candidates < 64:
        return 1
    if input_size <= 2048:
        # 35B (K=2048) has enough occupancy for tile=4 even for the larger
        # prefill buckets.  The tiled kernel keeps the same per-candidate
        # reduction order, so this is both faster and numerically stable.
        return 4
    if num_rows < 20:
        return 4
    if num_rows < 48:
        # K=5120 reaches the best occupancy with a wider tile for the common
        # BS16/32 mixed/decode shapes.
        return 8
    if num_rows < 80:
        return 4
    # At larger M, tile=8 becomes register-bound for the dense 27B head;
    # tile=2 is the stable low-register choice.
    return 2


@triton.jit
def _indexed_bf16_dot_kernel(
    HIDDEN,
    WEIGHT,
    INDICES,
    OUTPUT,
    HIDDEN_STRIDE_0: tl.constexpr,
    WEIGHT_STRIDE_0: tl.constexpr,
    INDEX_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    BLOCK_INPUT_SIZE: tl.constexpr,
):
    pair_id = tl.program_id(0)
    row = pair_id // NUM_CANDIDATES
    candidate = pair_id % NUM_CANDIDATES
    token_id = tl.load(INDICES + row * INDEX_STRIDE_0 + candidate)
    offsets = tl.arange(0, BLOCK_INPUT_SIZE)
    mask = offsets < INPUT_SIZE
    hidden = tl.load(
        HIDDEN + row * HIDDEN_STRIDE_0 + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        WEIGHT + token_id * WEIGHT_STRIDE_0 + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = tl.sum(hidden * weight, axis=0)
    tl.store(OUTPUT + row * OUTPUT_STRIDE_0 + candidate, value)


@triton.jit
def _tiled_indexed_bf16_dot_kernel(
    HIDDEN,
    WEIGHT,
    INDICES,
    OUTPUT,
    HIDDEN_STRIDE_0: tl.constexpr,
    WEIGHT_STRIDE_0: tl.constexpr,
    INDEX_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    BLOCK_INPUT_SIZE: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0)
    candidates = tl.program_id(1) * BLOCK_CANDIDATES + tl.arange(0, BLOCK_CANDIDATES)
    candidate_mask = candidates < NUM_CANDIDATES
    token_ids = tl.load(
        INDICES + row * INDEX_STRIDE_0 + candidates,
        mask=candidate_mask,
        other=0,
    )

    offsets = tl.arange(0, BLOCK_INPUT_SIZE)
    input_mask = offsets < INPUT_SIZE
    hidden = tl.load(
        HIDDEN + row * HIDDEN_STRIDE_0 + offsets,
        mask=input_mask,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        WEIGHT + token_ids[:, None] * WEIGHT_STRIDE_0 + offsets[None, :],
        mask=candidate_mask[:, None] & input_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    values = tl.sum(weights * hidden[None, :], axis=1)
    tl.store(
        OUTPUT + row * OUTPUT_STRIDE_0 + candidates,
        values,
        mask=candidate_mask,
    )


def indexed_bf16_dot(
    hidden_states: torch.Tensor,
    bf16_weight: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    candidate_tile: int | None = None,
    num_warps: int = 8,
) -> torch.Tensor:
    """Compute selected BF16 logits without materializing gathered weights."""
    assert hidden_states.ndim == 2
    assert bf16_weight.ndim == 2
    assert candidate_indices.ndim == 2
    assert hidden_states.shape[0] == candidate_indices.shape[0]
    assert hidden_states.shape[1] == bf16_weight.shape[1]
    assert hidden_states.dtype == torch.bfloat16
    assert bf16_weight.dtype == torch.bfloat16
    assert hidden_states.is_cuda
    assert bf16_weight.is_cuda
    assert candidate_indices.is_cuda
    assert hidden_states.is_contiguous()
    assert bf16_weight.is_contiguous()
    assert candidate_indices.is_contiguous()
    if num_warps not in (4, 8):
        raise ValueError(f"num_warps must be 4 or 8; got {num_warps}")

    output = torch.empty(
        candidate_indices.shape,
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    num_rows, num_candidates = candidate_indices.shape
    input_size = hidden_states.shape[1]
    block_input_size = triton.next_power_of_2(input_size)
    if candidate_tile is None:
        candidate_tile = _select_indexed_bf16_candidate_tile(
            num_rows,
            num_candidates,
            input_size,
        )
    if candidate_tile == 1:
        _indexed_bf16_dot_kernel[(num_rows * num_candidates,)](
            hidden_states,
            bf16_weight,
            candidate_indices,
            output,
            HIDDEN_STRIDE_0=hidden_states.stride(0),
            WEIGHT_STRIDE_0=bf16_weight.stride(0),
            INDEX_STRIDE_0=candidate_indices.stride(0),
            OUTPUT_STRIDE_0=output.stride(0),
            NUM_CANDIDATES=num_candidates,
            INPUT_SIZE=input_size,
            BLOCK_INPUT_SIZE=block_input_size,
            num_warps=num_warps,
        )
    else:
        if candidate_tile not in (2, 4, 8):
            raise ValueError(
                f"candidate_tile must be one of 1, 2, 4, or 8; got {candidate_tile}"
            )
        _tiled_indexed_bf16_dot_kernel[
            (num_rows, triton.cdiv(num_candidates, candidate_tile))
        ](
            hidden_states,
            bf16_weight,
            candidate_indices,
            output,
            HIDDEN_STRIDE_0=hidden_states.stride(0),
            WEIGHT_STRIDE_0=bf16_weight.stride(0),
            INDEX_STRIDE_0=candidate_indices.stride(0),
            OUTPUT_STRIDE_0=output.stride(0),
            NUM_CANDIDATES=num_candidates,
            INPUT_SIZE=input_size,
            BLOCK_INPUT_SIZE=block_input_size,
            BLOCK_CANDIDATES=candidate_tile,
            num_warps=num_warps,
        )
    return output


def select_lm_head_candidates(
    coarse_logits: torch.Tensor,
    candidates: int,
    *,
    use_flashinfer_topk: bool | None = None,
    format_name: str = "MXFP8",
) -> torch.Tensor:
    """Select an unsorted exact top-k set with the fastest available backend."""
    if use_flashinfer_topk is None:
        use_flashinfer_topk = True
    if use_flashinfer_topk and has_flashinfer():
        from flashinfer import top_k as flashinfer_top_k

        logger.info_once(
            "Hybrid %s lm-head is using FlashInfer exact unsorted top-k "
            "candidate selection; FlashInfer auto-dispatches its backend.",
            format_name,
        )
        _, candidate_indices = flashinfer_top_k(
            coarse_logits,
            candidates,
            sorted=False,
        )
        return candidate_indices
    return torch.topk(
        coarse_logits,
        candidates,
        dim=-1,
        sorted=False,
    ).indices


def autotune_row_buckets(max_rows: int) -> tuple[int, ...]:
    """Row shapes whose first runtime hit would trigger live FlashInfer tuning.

    The CUTLASS mm inside :meth:`HybridMxfp8LmHead.coarse_logits` keys its
    tactic cache with FlashInfer's hybrid num-tokens buckets (power-of-two up
    to 256, then 256 steps). A first runtime call on any new bucket costs a
    live autotune pass, so mirror FlashInfer's bucket list here and tune all
    of them during loading instead.
    """
    if max_rows <= 0:
        max_rows = _DEFAULT_AUTOTUNE_MAX_ROWS
    try:
        from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

        return get_hybrid_num_tokens_buckets(max_rows)
    except Exception:
        buckets = [b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256) if b <= max_rows]
        rows = 512
        while rows <= max_rows:
            buckets.append(rows)
            rows += 256
        if not buckets:
            buckets.append(max_rows)
        return tuple(sorted(set(buckets)))
