"""Exact unsorted top-k contract; no deterministic boundary-index policy."""
import torch


def assert_exact_unsorted_topk(coarse, reference_ids, actual_ids):
    assert coarse.ndim == reference_ids.ndim == actual_ids.ndim == 2
    assert reference_ids.shape == actual_ids.shape
    rows, width = actual_ids.shape
    assert rows == coarse.shape[0] and 0 < width <= coarse.shape[1]
    assert bool(torch.isfinite(coarse).all()), "nonfinite coarse logits"
    cutoff = coarse.topk(width, dim=-1).values[:, -1:]
    masks = []
    for ids in (reference_ids, actual_ids):
        assert ids.dtype in (torch.int32, torch.int64)
        assert bool(((ids >= 0) & (ids < coarse.shape[1])).all()), "invalid candidate ID"
        ordered = ids.sort(dim=-1).values
        assert not bool((ordered[:, 1:] == ordered[:, :-1]).any()), "duplicate candidate ID"
        selected = torch.zeros_like(coarse, dtype=torch.bool).scatter_(1, ids.long(), True)
        assert not bool(((coarse > cutoff) & ~selected).any()), "omitted strict top-k value"
        assert not bool((coarse.gather(1, ids.long()) < cutoff).any()), "selected below cutoff"
        masks.append(selected)
    difference = masks[0] ^ masks[1]
    assert not bool((difference & (coarse != cutoff)).any()), "non-boundary candidate substitution"
