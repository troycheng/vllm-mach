# SPDX-License-Identifier: Apache-2.0
"""Aligned GDN metadata preserves integer indices across CUDA graph replay."""

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec


@pytest.mark.parametrize("num_requests", [0, 1, 4])
@pytest.mark.parametrize("strided", [False, True])
def test_aligned_state_indices_accept_padded_cpu_rows(num_requests, strided):
    """Active lengths own the output shape; trailing table rows stay unused."""
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    spec = MambaSpec(
        block_size=16, shapes=((16, 128),), dtypes=(torch.bfloat16,),
        num_speculative_blocks=2,
    )
    lengths = torch.tensor([0, 99, 1, 99, 17, 99, 32, 99], dtype=torch.int32)
    lengths = lengths[::2] if strided else lengths[::2].contiguous()
    lengths = lengths[:num_requests]
    table = torch.arange((num_requests + 3) * 16, dtype=torch.int32).reshape(
        num_requests + 3, 16
    )
    if strided:
        table = table[:, ::2]
    starts = [0, 0, 1, 1][:num_requests]
    expected = torch.tensor(
        [[int(table[row, start + offset]) for offset in range(3)]
         for row, start in enumerate(starts)], dtype=table.dtype,
    ).reshape(num_requests, 3)
    result = mamba_get_block_table_tensor(table, lengths, spec, "align")
    assert result.shape == (num_requests, 3)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for mode in ("all", "none"):
        assert mamba_get_block_table_tensor(table, lengths, spec, mode) is table


def test_aligned_state_indices_reject_short_table_and_cross_device():
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    spec = MambaSpec(block_size=16, shapes=((16, 128),), dtypes=(torch.bfloat16,))
    table = torch.zeros((2, 8), dtype=torch.int32)
    with pytest.raises(AssertionError, match="fewer than"):
        mamba_get_block_table_tensor(table, torch.ones(3, dtype=torch.int32), spec, "align")
    # Meta is a different device without requiring a CUDA context in CPU CI.
    with pytest.raises(AssertionError, match="seq_lens is on"):
        mamba_get_block_table_tensor(table, torch.empty(2, device="meta"), spec, "align")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 320, 560])
@pytest.mark.parametrize("num_requests", [0, 1, 4, 16, 64, 65, 256])
@pytest.mark.parametrize("num_speculative_blocks", [0, 2])
@pytest.mark.parametrize("strided", [False, True])
def test_aligned_state_indices_match_cpu_and_replay(
    block_size, num_requests, num_speculative_blocks, strided
):
    """Preserve padded rows, block boundaries, strides and graph input updates."""
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    spec = MambaSpec(
        block_size=block_size,
        shapes=((16, 128),),
        dtypes=(torch.bfloat16,),
        num_speculative_blocks=num_speculative_blocks,
    )
    lengths = torch.tensor(
        [0, 1, block_size - 1, block_size, block_size + 1, 2 * block_size],
        device="cuda",
        dtype=torch.int32,
    ).repeat((2 * num_requests + 5) // 6)[: 2 * num_requests]
    lengths = lengths[::2] if strided else lengths[:num_requests]
    table = torch.arange(
        (num_requests + 3) * 128, device="cuda", dtype=torch.int32
    ).reshape(num_requests + 3, 128)
    if strided:
        table = table[:, ::2]

    def select(table, lengths):
        return mamba_get_block_table_tensor(table, lengths, spec, "align")

    torch.testing.assert_close(
        select(table, lengths).cpu(),
        select(table.cpu(), lengths.cpu()),
        rtol=0,
        atol=0,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = select(table, lengths)
    for increment in (1, block_size):
        lengths.add_(increment)
        graph.replay()
        torch.testing.assert_close(
            result.cpu(), select(table.cpu(), lengths.cpu()), rtol=0, atol=0
        )
    for mode in ("all", "none"):
        assert mamba_get_block_table_tensor(table, lengths, spec, mode) is table
