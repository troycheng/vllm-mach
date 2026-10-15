"""Sampling admission must preserve seeded/full-logit requests and shard semantics."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_mach.mxfp6 import sampling


def fixture_sampler(k=4, p=0.9, temperature=0.7, penalty=0.0):
    def array(value):
        return NS(np=np.full(2, value))

    states = NS(
        temperature=array(temperature),
        top_k=array(k),
        top_p=array(p),
        min_p=array(0.0),
        vocab_size=128,
        any_explicit_seed=lambda _: False,
    )
    penalties = NS(
        use_penalty=np.full(2, penalty != 0),
        repetition_penalty=array(1.0),
        frequency_penalty=array(0.0),
        presence_penalty=array(penalty),
    )
    sampler = NS(
        sampling_states=states,
        penalties_state=penalties,
        compute_nans=False,
        use_fp64_gumbel=False,
        return_sampling_mask=False,
        trace_replay_state=None,
        get_logprobs_dims=lambda _: None,
        logit_bias_state=NS(use_logit_bias=np.zeros(2, bool)),
        bad_words_state=NS(num_bad_words=array(0)),
        thinking_budget_state=NS(enabled=False),
    )
    return sampler, NS(idx_mapping_np=np.arange(2))


@pytest.mark.parametrize(
    "mode,k,p,penalty",
    [
        ("topk", 4, 0.9, 0),
        ("topk", 4, 0.9, 1.5),
        ("topk", 4, 0.9, -1.5),
        ("topk", 20, 1.0, 0),
        ("topk", 64, 1.0, 1.5),
    ],
)
def test_uniform_unseeded_stochastic_modes(mode, k, p, penalty):
    sampler, batch = fixture_sampler(k, p, penalty=penalty)
    assert sampling.sampling_params(sampler, batch) == (mode, k, p, 0.7, penalty != 0)


@pytest.mark.parametrize(
    "reason",
    [
        "seed",
        "logprobs",
        "grammar_mask",
        "trace",
        "nan",
        "fp64",
        "bias",
        "bad_words",
        "thinking",
        "repetition",
        "frequency",
        "min_p",
        "heterogeneous",
        "greedy",
        "top_p_only",
        "large_k",
        "full_presence",
    ],
)
def test_full_logit_and_seeded_requests_fall_back(reason):
    s, b = fixture_sampler()
    if reason == "seed":
        s.sampling_states.any_explicit_seed = lambda _: True
    elif reason == "logprobs":
        s.get_logprobs_dims = lambda _: (1, 0)
    elif reason == "grammar_mask":
        s.return_sampling_mask = True
    elif reason == "trace":
        s.trace_replay_state = object()
    elif reason == "nan":
        s.compute_nans = True
    elif reason == "fp64":
        s.use_fp64_gumbel = True
    elif reason == "bias":
        s.logit_bias_state.use_logit_bias[0] = True
    elif reason == "bad_words":
        s.bad_words_state.num_bad_words.np[0] = 1
    elif reason == "thinking":
        s.thinking_budget_state = NS(enabled=True, use_thinking_budget=np.ones(2, bool))
    elif reason in ("repetition", "frequency"):
        s.penalties_state.use_penalty[:] = True
        getattr(s.penalties_state, reason + "_penalty").np[:] = 2
    elif reason == "min_p":
        s.sampling_states.min_p.np[0] = 0.1
    elif reason == "heterogeneous":
        s.sampling_states.temperature.np[0] = 0.8
    elif reason == "greedy":
        s.sampling_states.temperature.np[:] = 0
    elif reason == "top_p_only":
        s.sampling_states.top_k.np[:] = 128
    elif reason == "large_k":
        s.sampling_states.top_k.np[:] = 65
    elif reason == "full_presence":
        s.sampling_states.top_k.np[:] = 128
        s.sampling_states.top_p.np[:] = 1
        s.penalties_state.use_penalty[:] = True
    assert sampling.sampling_params(s, b) is None


def make_head(vocab=16):
    shard = NS(
        org_vocab_start_index=0,
        org_vocab_end_index=vocab,
        num_org_vocab_padding=0,
        num_added_elements_padded=0,
        num_elements_padded=vocab,
        num_org_elements=vocab,
        num_org_elements_padded=vocab,
    )
    weight = torch.arange(vocab * 4, dtype=torch.float32).view(vocab, 4) / 100
    return NS(
        shard_indices=shard,
        weight=weight,
        tp_size=1,
        quant_method=NS(apply=lambda head, x, bias=None: x @ head.weight.T),
    )


def test_presence_is_applied_before_candidate_selection():
    head = make_head()
    proxy = sampling.SamplingHead(NS(scale=1.0, soft_cap=None))
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    counts = torch.zeros((2, 16), dtype=torch.int32)
    counts[1, 15] = 1
    values, ids = proxy.get_topk_candidates(
        head,
        hidden,
        3,
        1.0,
        0.5,
        presence_penalties=torch.tensor([5.0]),
        output_token_counts=counts,
        presence_request_indices=torch.tensor([1]),
    )
    full = hidden @ head.weight.T
    full[0, 15] -= 5
    expected = full.div(0.5).topk(3)
    torch.testing.assert_close(values, expected.values)
    assert torch.equal(ids, expected.indices)


def test_top_p_uses_global_candidates_after_tp_merge():
    proxy = sampling.SamplingHead(NS(scale=1.0, soft_cap=None))
    pairs = torch.tensor([[[3.0, 100.0], [1.0, 101.0], [2.0, 200.0], [0.0, 201.0]]])
    values, ids = proxy._select_from_compact_topk_pairs(pairs, top_k=3, top_p=0.7)
    assert ids.tolist() == [[100, 200, 101]]
    assert torch.isfinite(values).tolist() == [[True, True, False]]


@pytest.mark.parametrize("penalty", [-5.0, 5.0])
def test_hybrid_presence_changes_candidates_before_bf16_refinement(penalty):
    head = make_head()
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    full = hidden @ head.weight.T
    seen = []

    def select(coarse, *, top_k):
        seen.append(top_k)
        return coarse.topk(2).indices

    state = NS(
        can_use=lambda *a, **kw: True,
        candidate_count_for_topk=lambda k: 2,
        coarse_logits=lambda *a: full.clone(),
        select_candidates=select,
        refine_logits=lambda x, w, ids, bias: full.gather(1, ids),
    )
    proxy = sampling.SamplingHead(NS(scale=1.0, soft_cap=None), state)
    counts = torch.zeros((1, 16), dtype=torch.int32)
    counts[0, 15] = 1
    values, ids = proxy.get_topk_candidates(
        head,
        hidden,
        2,
        1.0,
        0.5,
        presence_penalties=torch.tensor([penalty]),
        output_token_counts=counts,
        presence_request_indices=torch.tensor([0]),
    )
    full[0, 15] -= penalty
    expected = (full / 0.5).topk(2)
    assert seen == [2]
    assert torch.equal(ids, expected.indices)
    torch.testing.assert_close(values, expected.values)


def test_mixed_batch_delegates_original_sampler_once(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    calls = []
    original_output = object()

    def original(self, hidden_states, input_batch, grammar_output):
        calls.append((hidden_states, input_batch, grammar_output))
        return original_output

    monkeypatch.setattr(GPUModelRunner, "sample", original)
    monkeypatch.setenv(sampling.ENV, "1")
    sampling.install_sampling_hook()
    sampling.install_sampling_hook()
    runner = NS(lora_config=None, speculative_config=None, batch_sharder=None)
    batch = NS(num_draft_tokens=0, is_prefilling_np=np.array([False, True]))
    assert GPUModelRunner.sample(runner, "hidden", batch, None) is original_output
    assert calls == [("hidden", batch, None)]


@pytest.mark.parametrize("vocab", [32, 128])
@pytest.mark.parametrize("k", [-1, 0, None])
@pytest.mark.parametrize("p,penalty", [(1.0, 0.0), (1.0, 1.5), (0.95, 0.0)])
def test_unbounded_sampling_always_delegates_original_sampler(
    monkeypatch, vocab, k, p, penalty
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    sampler, batch = fixture_sampler(vocab if k is None else k, p, penalty=penalty)
    sampler.sampling_states.vocab_size = vocab
    batch.num_draft_tokens = 0
    batch.is_prefilling_np = np.zeros(2, bool)
    runner = NS(
        lora_config=None, speculative_config=None, batch_sharder=None, sampler=sampler
    )
    calls = []
    original_output = object()

    def original(self, hidden_states, input_batch, grammar_output):
        calls.append((hidden_states, input_batch, grammar_output))
        return original_output

    def unexpected_proxy(*args, **kwargs):
        pytest.fail("unbounded sampling must not prepare an optimized head")

    monkeypatch.setattr(GPUModelRunner, "sample", original)
    monkeypatch.setattr(sampling, "SamplingHead", unexpected_proxy)
    monkeypatch.setenv(sampling.ENV, "1")
    sampling.install_sampling_hook()
    assert GPUModelRunner.sample(runner, "hidden", batch, None) is original_output
    assert calls == [("hidden", batch, None)]


def test_disabled_hook_does_not_attempt_optimized_sampling(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original_output = object()
    monkeypatch.setattr(GPUModelRunner, "sample", lambda *args: original_output)
    monkeypatch.setattr(
        sampling, "try_sample", lambda *args: pytest.fail("sampling was disabled")
    )
    monkeypatch.setenv(sampling.ENV, "0")
    sampling.install_sampling_hook()
    assert GPUModelRunner.sample(None, None, None, None) is original_output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_sampling_matches_its_top_p_distribution():
    from vllm_mach.mxfp6.sampling_triton import sample_from_compact_topk_pairs_triton

    n = 20000
    pairs = (
        torch.tensor(
            [[[2.0, 7.0], [1.0, 11.0], [0.0, 15.0], [-1.0, 19.0]]], device="cuda"
        )
        .expand(n, -1, -1)
        .contiguous()
    )
    seeds = torch.arange(n, device="cuda", dtype=torch.int64) + 20261015
    ids = sample_from_compact_topk_pairs_triton(pairs, top_k=3, top_p=0.8, seeds=seeds)
    expected = torch.tensor([2.0, 1.0], device="cuda").softmax(0)
    observed = torch.stack([(ids == i).float().mean() for i in (7, 11)])
    assert ((ids == 7) | (ids == 11)).all()
    torch.testing.assert_close(observed, expected, atol=0.015, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shard_gumbel_uses_global_token_random_offsets():
    from vllm_mach.mxfp6.sampling_triton import (
        sample_full_vocab_from_shard_triton as sample,
    )

    torch.manual_seed(15)
    logits = torch.randn((32, 256), device="cuda")
    seeds = torch.arange(32, device="cuda", dtype=torch.int64) + 123
    values, ids = sample(
        logits, vocab_start=0, active_vocab_size=256, seeds=seeds, temperature=0.7
    )
    left, lids = sample(
        logits[:, :128].contiguous(),
        vocab_start=0,
        active_vocab_size=128,
        seeds=seeds,
        temperature=0.7,
    )
    right, rids = sample(
        logits[:, 128:].contiguous(),
        vocab_start=128,
        active_vocab_size=128,
        seeds=seeds,
        temperature=0.7,
    )
    assert torch.equal(ids, torch.where(left >= right, lids, rids))
    torch.testing.assert_close(values, torch.maximum(left, right), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_sampling_matches_softmax_distribution():
    from vllm_mach.mxfp6.sampling_triton import sample_full_vocab_from_shard_triton

    n = 20000
    logits = (
        torch.tensor([[2.0, 1.0, 0.0, -1.0]], device="cuda").expand(n, -1).contiguous()
    )
    _, ids = sample_full_vocab_from_shard_triton(
        logits,
        vocab_start=0,
        active_vocab_size=4,
        seeds=torch.arange(n, device="cuda", dtype=torch.int64) + 20261015,
        temperature=0.7,
    )
    observed = torch.stack([(ids == i).float().mean() for i in range(4)])
    torch.testing.assert_close(
        observed, (logits[0] / 0.7).softmax(0), atol=0.015, rtol=0
    )
