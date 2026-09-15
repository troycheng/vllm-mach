# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm_mach.mxfp6.hybrid_nvfp4_lm_head as hybrid_nvfp4
from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import (
    HybridNvfp4LmHead,
    prepare_hybrid_nvfp4_lm_head,
    release_hybrid_nvfp4_lm_head,
)


def _state(**kwargs) -> HybridNvfp4LmHead:
    defaults = dict(
        weight=torch.empty((8, 2), dtype=torch.uint8),
        scale=torch.empty((128, 4), dtype=torch.uint8),
        global_scale=torch.tensor(1.0),
        input_size=4,
        output_size=8,
        candidates=4,
        max_rows=32,
    )
    defaults.update(kwargs)
    return HybridNvfp4LmHead(**defaults)


def test_nvfp4_global_scale_zero_is_finite() -> None:
    scale = hybrid_nvfp4._global_scale(torch.zeros((2, 4), dtype=torch.bfloat16))
    assert torch.isfinite(scale)
    assert scale.item() == 1.0


def test_nvfp4_global_scale_matches_fp32_reference_for_finite_bf16() -> None:
    tensor = torch.tensor(
        [[-3.5, 1.25, 0.0, 2.0], [4.0, -0.5, 1.0, -2.5]],
        dtype=torch.bfloat16,
    )
    expected_max = tensor.float().abs().nan_to_num().amax()
    expected = torch.where(
        expected_max > 0,
        expected_max.clamp_min(1.0e-12).reciprocal() * 448.0 * 6.0,
        torch.ones_like(expected_max),
    )
    torch.testing.assert_close(hybrid_nvfp4._global_scale(tensor), expected)


def test_nvfp4_can_use_records_compact_path_failure() -> None:
    state = _state(candidates=2)
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    weight = torch.ones((8, 4), dtype=torch.bfloat16)

    assert not state.can_use(
        hidden,
        bf16_weight=weight,
        active_vocab_size=8,
        top_k=1,
    )
    assert state.can_use_failure_counts == {"hidden_not_cuda": 1}


def test_nvfp4_coarse_gemm_uses_b12x_contract(monkeypatch) -> None:
    state = _state()
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        hybrid_nvfp4,
        "_global_scale",
        lambda value: torch.tensor(2.0),
    )

    def fake_quantize(value: torch.Tensor, global_scale: torch.Tensor):
        calls["quantize"] = (tuple(value.shape), float(global_scale))
        return (
            torch.empty((value.shape[0], 2), dtype=torch.uint8),
            torch.empty((128, 4), dtype=torch.uint8),
        )

    def fake_mm(*args, **kwargs):
        calls["mm_args"] = args
        calls["mm_kwargs"] = kwargs
        return torch.zeros((2, 8), dtype=torch.bfloat16)

    monkeypatch.setattr(
        hybrid_nvfp4,
        "flashinfer_nvfp4_quantize_128x4",
        fake_quantize,
    )
    monkeypatch.setattr(hybrid_nvfp4, "flashinfer_scaled_fp4_mm", fake_mm)

    output = state.coarse_logits(hidden, None)

    assert output.shape == (2, 8)
    assert calls["quantize"] == ((2, 4), 2.0)
    assert calls["mm_kwargs"] == {
        "alpha": torch.tensor(0.5),
        "out_dtype": torch.bfloat16,
        "backend": "b12x",
        "block_size": 16,
        "use_nvfp4": True,
    }


def test_nvfp4_shared_state_attaches_without_requantizing(monkeypatch) -> None:
    weight = torch.nn.Parameter(torch.empty((256, 32), dtype=torch.bfloat16))
    state = _state(
        weight=torch.empty((256, 16), dtype=torch.uint8),
        scale=torch.empty((256, 2), dtype=torch.uint8),
        input_size=32,
        output_size=256,
        candidates=40,
        max_rows=64,
    )
    weight._hybrid_nvfp4_lm_head_shared_state = state
    layer = torch.nn.Module()
    layer.weight = weight

    monkeypatch.setattr(hybrid_nvfp4, "has_flashinfer", lambda: False)

    assert prepare_hybrid_nvfp4_lm_head(layer, candidates=40)
    assert layer._hybrid_nvfp4_lm_head_state is state
    assert layer._hybrid_nvfp4_lm_head_weight is state.weight
    assert layer._hybrid_nvfp4_lm_head_scale is state.scale


def test_release_nvfp4_lm_head_drops_registered_buffers() -> None:
    layer = torch.nn.Module()
    weight = torch.empty((8, 2), dtype=torch.uint8)
    scale = torch.empty((128, 4), dtype=torch.uint8)
    global_scale = torch.tensor(1.0)
    layer.register_buffer("_hybrid_nvfp4_lm_head_weight", weight, persistent=False)
    layer.register_buffer("_hybrid_nvfp4_lm_head_scale", scale, persistent=False)
    layer.register_buffer(
        "_hybrid_nvfp4_lm_head_global_scale",
        global_scale,
        persistent=False,
    )
    layer._hybrid_nvfp4_lm_head_state = object()

    released = release_hybrid_nvfp4_lm_head(layer)

    assert released == weight.nbytes + scale.nbytes + global_scale.nbytes
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_state")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_weight")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_scale")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_global_scale")


def test_nvfp4_candidate_refinement_and_graph_replay(monkeypatch) -> None:
    """All-candidate refinement preserves BF16 logits, including under replay."""
    import pytest
    from vllm.platforms import current_platform

    if not torch.cuda.is_available() or not current_platform.has_device_capability(120):
        pytest.skip("NVFP4 b12x requires SM120")
    monkeypatch.setenv("VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS", "32")
    torch.manual_seed(42)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
    )
    assert prepare_hybrid_nvfp4_lm_head(layer, candidates=256)
    state = hybrid_nvfp4.get_hybrid_nvfp4_lm_head(layer)
    hidden = torch.randn((32, 512), device="cuda", dtype=torch.bfloat16)

    def run():
        coarse = state.coarse_logits(hidden, None)
        indices = state.select_candidates(coarse)
        return indices, state.refine_logits(hidden, layer.weight, indices, None)

    indices, values = run()
    expected = torch.nn.functional.linear(hidden, layer.weight).gather(
        1, indices.long()
    )
    torch.testing.assert_close(values, expected, atol=0.25, rtol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices, values = run()
    hidden.normal_()
    graph.replay()
    expected = torch.nn.functional.linear(hidden, layer.weight).gather(
        1, indices.long()
    )
    torch.testing.assert_close(values, expected, atol=0.25, rtol=0.01)


def test_compact_head_masks_padding_and_preserves_full_logits(
    monkeypatch, default_vllm_config
) -> None:
    """Padded vocab must never win; logprob callers still get original logits."""
    from types import SimpleNamespace

    import pytest
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.platforms import current_platform

    if not torch.cuda.is_available() or not current_platform.has_device_capability(120):
        pytest.skip("NVFP4 b12x requires SM120")
    monkeypatch.setenv("VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS", "32")
    torch.manual_seed(19)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
    )
    layer.tp_size = 1
    layer.quant_method = UnquantizedLinearMethod()
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=192,
        num_org_vocab_padding=64,
        num_added_elements_padded=0,
    )
    assert prepare_hybrid_nvfp4_lm_head(layer, candidates=192)
    processor = LogitsProcessor(vocab_size=192)
    hidden = torch.randn((32, 512), device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
    bias[192:] = 10000
    dense = torch.nn.functional.linear(hidden, layer.weight, bias)[:, :192]
    torch.testing.assert_close(processor(layer, hidden, bias), dense, rtol=0, atol=0)
    tokens = processor.get_top_tokens(layer, hidden, bias)
    assert (tokens < 192).all()
    # Compare against the same FP32 reduction used by candidate refinement.
    exact = torch.einsum("mk,nk->mn", hidden.float(), layer.weight[:192].float())
    torch.testing.assert_close(tokens, exact.bfloat16().argmax(-1))
