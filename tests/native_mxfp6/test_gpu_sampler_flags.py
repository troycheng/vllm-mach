# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, id="penalty"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), True, id="bad-words"),
        pytest.param(SamplingParams(temperature=0.7), True, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, prompt_len=1, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, 1, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, 1, SamplingParams())

    assert not sampler.needs_logits_processing[3]


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(2, 1, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (SamplingParams(temperature=0), True),
        (SamplingParams(temperature=0, logprobs=0), False),
        (SamplingParams(temperature=0, logprob_token_ids=[1]), False),
        (SamplingParams(temperature=0, logit_bias={1: 2}), False),
        (SamplingParams(temperature=0, frequency_penalty=0.1), False),
        (SamplingParams(temperature=0, min_tokens=2, stop_token_ids=[2]), False),
        (SamplingParams(temperature=0, allowed_token_ids=[1, 2]), False),
        (SamplingParams(temperature=0, _bad_words_token_ids=[[1]]), False),
        (SamplingParams(temperature=0, thinking_token_budget=3), False),
        (SamplingParams(temperature=0.7, top_k=20), False),
    ],
)
def test_compact_greedy_requires_unmodified_logits(params, expected):
    from types import SimpleNamespace

    sampler = _make_sampler()
    sampler.add_request(0, 1, params)
    sampler.add_request(2, 1, SamplingParams.for_sampler_warmup())
    batch = SimpleNamespace(
        num_reqs=1, num_draft_tokens=0, idx_mapping_np=np.array([0])
    )
    assert sampler.can_sample_compact_greedy(batch) == expected


@pytest.mark.parametrize(
    "reason", ["compute_nans", "return_sampling_mask", "draft", "empty", "trace"]
)
def test_compact_greedy_falls_back_for_diagnostics_and_speculation(reason):
    from types import SimpleNamespace

    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0))
    batch = SimpleNamespace(
        num_reqs=1, num_draft_tokens=0, idx_mapping_np=np.array([0])
    )
    if reason == "draft":
        batch.num_draft_tokens = 1
    elif reason == "empty":
        batch.num_reqs = 0
    elif reason == "trace":
        sampler.trace_replay_state = object()
    else:
        setattr(sampler, reason, True)
    assert not sampler.can_sample_compact_greedy(batch)
