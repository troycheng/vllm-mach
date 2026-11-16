"""Tie flexibility never permits dropping strict top-k values or invalid IDs."""
import pytest
import torch
from head_topk_contract import assert_exact_unsorted_topk


def test_unsorted_topk_accepts_only_boundary_tie_substitution():
    coarse = torch.tensor([[9., 8., 8., 8., 1.]])
    assert_exact_unsorted_topk(coarse, torch.tensor([[0, 1, 2]]), torch.tensor([[3, 0, 2]]))
    assert_exact_unsorted_topk(coarse, torch.tensor([[2, 0, 1]]), torch.tensor([[1, 2, 0]]))


@pytest.mark.parametrize("ids", [[0, 1, 4], [1, 2, 3], [0, 1, 1], [0, 1, 5], [0, 1, -1]])
def test_unsorted_topk_rejects_invalid_selection(ids):
    coarse = torch.tensor([[9., 8., 8., 8., 1.]])
    with pytest.raises(AssertionError):
        assert_exact_unsorted_topk(coarse, torch.tensor([[0, 1, 2]]), torch.tensor([ids]))
