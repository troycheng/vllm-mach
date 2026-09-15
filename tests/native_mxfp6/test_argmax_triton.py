# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_mach.mxfp6.argmax_triton import (
    indexed_argmax_triton,
    local_argmax_triton,
)


@pytest.mark.parametrize(
    ("active_vocab_size", "vocab_start"),
    [
        (248320, 0),
        (124160, 124160),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_argmax_all_nan_row_stays_in_active_vocab(
    active_vocab_size: int,
    vocab_start: int,
) -> None:
    logits = torch.full(
        (1, active_vocab_size),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )

    values, token_ids = local_argmax_triton(
        logits,
        vocab_start=vocab_start,
        active_vocab_size=active_vocab_size,
    )

    assert torch.isneginf(values).all()
    assert token_ids.item() == vocab_start
    assert token_ids.item() < vocab_start + active_vocab_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexed_argmax_uses_minimum_token_id_for_ties() -> None:
    values = torch.tensor(
        [[1.0, 3.0, 3.0, 2.0], [4.0, 4.0, 3.0, 2.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    token_ids = torch.tensor(
        [[8, 7, 2, 1], [9, 4, 3, 2]],
        dtype=torch.int64,
        device="cuda",
    )

    max_values, max_token_ids = indexed_argmax_triton(
        values, token_ids, index_offset=100
    )

    assert max_values.tolist() == [3.0, 4.0]
    assert max_token_ids.tolist() == [102, 104]
