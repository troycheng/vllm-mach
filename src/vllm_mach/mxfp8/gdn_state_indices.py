# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused aligned Mamba state block selection, shared with Kimi-K3."""

from functools import cache

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import MambaSpec


@cache
def _metadata_launch_pdl() -> bool:
    return current_platform.is_arch_support_pdl()


@triton.jit(do_not_specialize=["num_requests"])
def _get_aligned_state_indices_kernel(
    block_table_ptr,
    seq_lens_ptr,
    state_indices_ptr,
    block_table_stride_0: tl.constexpr,
    block_table_stride_1: tl.constexpr,
    seq_lens_stride: tl.constexpr,
    state_indices_stride_0: tl.constexpr,
    state_indices_stride_1: tl.constexpr,
    num_requests,
    CACHE_BLOCK_SIZE: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
    BLOCK_STATE_SLOTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid_row = rows < num_requests
    seq_lens = tl.load(
        seq_lens_ptr + rows * seq_lens_stride,
        mask=valid_row,
        other=1,
    )
    # Triton truncates signed division toward zero, unlike PyTorch floor
    # division. Clamping makes both semantics equivalent for seq_lens <= 0.
    first_state_slot = tl.maximum((seq_lens - 1) // CACHE_BLOCK_SIZE, 0)

    state_slots = tl.arange(0, BLOCK_STATE_SLOTS)
    valid_state_slot = state_slots < NUM_STATE_SLOTS
    state_indices = tl.load(
        block_table_ptr
        + rows[:, None] * block_table_stride_0
        + (first_state_slot[:, None] + state_slots[None, :]) * block_table_stride_1,
        mask=valid_row[:, None] & valid_state_slot[None, :],
    )
    tl.store(
        state_indices_ptr
        + rows[:, None] * state_indices_stride_0
        + state_slots[None, :] * state_indices_stride_1,
        state_indices,
        mask=valid_row[:, None] & valid_state_slot[None, :],
    )


def get_aligned_mamba_block_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: MambaSpec,
) -> torch.Tensor:
    assert block_table.is_cuda and seq_lens.is_cuda
    num_requests = seq_lens.numel()
    num_state_slots = 1 + kv_cache_spec.num_speculative_blocks
    state_indices = torch.empty(
        (num_requests, num_state_slots),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    if num_requests == 0:
        return state_indices
    BLOCK_ROWS = 32
    grid = (triton.cdiv(num_requests, BLOCK_ROWS),)
    _get_aligned_state_indices_kernel[grid](
        block_table,
        seq_lens,
        state_indices,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens.stride(0),
        state_indices.stride(0),
        state_indices.stride(1),
        num_requests,
        CACHE_BLOCK_SIZE=kv_cache_spec.block_size,
        NUM_STATE_SLOTS=num_state_slots,
        BLOCK_STATE_SLOTS=triton.next_power_of_2(num_state_slots),
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=1,
        launch_pdl=_metadata_launch_pdl(),
    )
    return state_indices
