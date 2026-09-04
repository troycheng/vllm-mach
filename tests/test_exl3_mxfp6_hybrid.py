from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_mach.exl3 import fused_mlp
from vllm_mach.exl3 import mxfp6_hybrid as hybrid
from vllm_mach.exl3.dense_adapter import Exl3Config, Exl3LinearMethod


class _TensorMap:
    def __init__(self, tensors):
        self.exl3_tensors = tensors


def _layer(prefix: str, shard_ids, output_sizes) -> SimpleNamespace:
    trellis = {
        shard_id: torch.zeros((8, 1, 96), dtype=torch.int16) for shard_id in shard_ids
    }
    signs = {
        shard_id: torch.ones(128 if name == "suh" else 16)
        for shard_id in shard_ids
        for name in ("suh",)
    }
    svh = {shard_id: torch.ones(16) for shard_id in shard_ids}
    mcg = {shard_id: torch.empty(0) for shard_id in shard_ids}
    return SimpleNamespace(
        prefix=prefix,
        exl3_shard_ids=list(shard_ids),
        exl3_output_partition_sizes=list(output_sizes),
        trellis=_TensorMap(trellis),
        suh=_TensorMap(signs),
        svh=_TensorMap(svh),
        mcg=_TensorMap(mcg),
        mul1=_TensorMap({}),
    )


class _Extension:
    @staticmethod
    def reconstruct_had_slice(
        output,
        trellis,
        suh,
        svh,
        bits,
        has_mcg,
        has_mul1,
        start,
    ):
        del trellis, suh, svh, bits, has_mcg, has_mul1, start
        output.copy_(
            torch.arange(output.numel(), dtype=output.dtype).reshape_as(output)
        )


class _Runtime:
    calls = 0
    fail_after = None

    @classmethod
    def quantize_mxfp6(cls, weight):
        cls.calls += 1
        if cls.fail_after is not None and cls.calls > cls.fail_after:
            raise RuntimeError("injected conversion failure")
        rows, k = weight.shape
        return SimpleNamespace(
            values=torch.empty((rows, k * 3 // 4), dtype=torch.uint8),
            scales=torch.empty((rows, k // 32), dtype=torch.uint8),
            rows=rows,
            k=k,
        )


@pytest.fixture(autouse=True)
def _reset_runtime():
    _Runtime.calls = 0
    _Runtime.fail_after = None


def test_profile_routes_only_the_validated_projection_families(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    all_rows = (
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.3.self_attn.o_proj",
    )
    prefill = (
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.3.self_attn.qkv_proj",
    )
    excluded = (
        "model.layers.3.self_attn.q_proj",
        "model.layers.0.linear_attn.in_proj_a",
        "model.embed_tokens",
        "lm_head",
    )
    assert all(
        hybrid.route_for_prefix(prefix) is hybrid.HybridRoute.ALL_ROWS
        for prefix in all_rows
    )
    assert all(
        hybrid.route_for_prefix(prefix) is hybrid.HybridRoute.PREFILL_ONLY
        for prefix in prefill
    )
    assert all(hybrid.route_for_prefix(prefix) is None for prefix in excluded)


def test_unknown_profile_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, "qwen-anything")
    with pytest.raises(ValueError, match="only accepts"):
        hybrid.route_for_prefix("model.layers.0.mlp.down_proj")


def test_explicit_profile_fails_closed_without_mxfp6_runtime(monkeypatch) -> None:
    def missing_runtime(_name: str) -> str:
        raise hybrid.PackageNotFoundError

    monkeypatch.setattr(hybrid, "version", missing_runtime)
    with pytest.raises(RuntimeError, match="requires mxfp6-sm120==0.2.1"):
        hybrid._load_runtime(torch.device("cpu"))


def test_explicit_profile_validates_exact_model_geometry(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    with pytest.raises(ValueError, match="requires a model configuration"):
        hybrid.validate_profile_model(None)
    valid = SimpleNamespace(num_hidden_layers=64, intermediate_size=17408)
    hybrid.validate_profile_model(valid)
    with pytest.raises(ValueError, match="num_hidden_layers=32"):
        hybrid.validate_profile_model(
            SimpleNamespace(num_hidden_layers=32, intermediate_size=17408)
        )


def test_gate_up_conversion_is_merged_and_committed_atomically(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    monkeypatch.setattr(hybrid, "_load_runtime", lambda device: _Runtime)
    layer = _layer("model.layers.0.mlp.gate_up_proj", [0, 1], [8, 8])

    state = hybrid.prepare_layer(layer, _Extension())

    assert state is not None
    assert state.route is hybrid.HybridRoute.ALL_ROWS
    assert state.weights == {}
    assert state.merged_weight is not None
    assert (state.merged_weight.rows, state.merged_weight.k) == (16, 128)
    assert _Runtime.calls == 1
    assert layer.trellis.exl3_tensors == {}
    assert hybrid.prepare_layer(layer, _Extension()) is state
    assert _Runtime.calls == 1


def test_failed_conversion_keeps_original_exl3_state(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    monkeypatch.setattr(hybrid, "_load_runtime", lambda device: _Runtime)
    _Runtime.fail_after = 1
    layer = _layer("model.layers.0.mlp.down_proj", [0, 1], [8, 8])
    original = dict(layer.trellis.exl3_tensors)

    with pytest.raises(RuntimeError, match="injected conversion failure"):
        hybrid.prepare_layer(layer, _Extension())

    assert not hasattr(layer, "_mach_exl3_mxfp6")
    assert layer.trellis.exl3_tensors == original


def test_prefill_route_keeps_trellis_and_switches_at_128(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    monkeypatch.setattr(hybrid, "_load_runtime", lambda device: _Runtime)
    layer = _layer(
        "model.layers.3.self_attn.qkv_proj",
        ["q", "k", "v"],
        [8, 8, 8],
    )

    state = hybrid.prepare_layer(layer, _Extension())

    assert state is not None
    assert state.route is hybrid.HybridRoute.PREFILL_ONLY
    assert set(layer.trellis.exl3_tensors) == {"q", "k", "v"}
    assert hybrid.state_for_rows(layer, 127) is None
    assert hybrid.state_for_rows(layer, 128) is state


def test_apply_preserves_shape_dtype_bias_and_row_boundary(monkeypatch) -> None:
    weights = {
        "q": hybrid.HybridPackedWeight(torch.empty(0), torch.empty(0), rows=8, k=128),
        "k": hybrid.HybridPackedWeight(torch.empty(0), torch.empty(0), rows=8, k=128),
    }
    layer = SimpleNamespace(
        prefix="model.layers.3.self_attn.qkv_proj",
        exl3_shard_ids=["q", "k"],
        _mach_exl3_mxfp6=hybrid.HybridState(
            route=hybrid.HybridRoute.PREFILL_ONLY,
            weights=weights,
        ),
    )
    seen_dtypes: list[torch.dtype] = []

    def fake_hybrid(x, weight):
        seen_dtypes.append(x.dtype)
        return torch.full((x.shape[0], weight.rows), 2, dtype=torch.bfloat16)

    monkeypatch.setattr(hybrid, "apply_weight", fake_hybrid)
    monkeypatch.setattr(
        Exl3LinearMethod,
        "_apply_one",
        staticmethod(
            lambda layer, x, shard_id: torch.zeros((x.shape[0], 8), dtype=torch.float16)
        ),
    )
    method = Exl3LinearMethod(Exl3Config())

    decode = method.apply(layer, torch.ones((1, 127, 128), dtype=torch.float32))
    prefill = method.apply(
        layer,
        torch.ones((1, 128, 128), dtype=torch.float32),
        bias=torch.ones(16),
    )

    assert decode.shape == (1, 127, 16)
    assert torch.count_nonzero(decode) == 0
    assert prefill.shape == (1, 128, 16)
    assert prefill.dtype is torch.float32
    assert torch.all(prefill == 3)
    assert seen_dtypes == [torch.bfloat16, torch.bfloat16]


def test_hybrid_native_operator_matches_runtime_reference() -> None:
    mxfp6 = pytest.importorskip("mxfp6")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    if not mxfp6.is_available():
        pytest.skip("mxfp6-sm120 is unavailable")

    weight = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    packed = mxfp6.quantize_mxfp6(weight)
    hybrid_weight = hybrid.HybridPackedWeight(
        values=packed.values,
        scales=packed.scales,
        rows=packed.rows,
        k=packed.k,
    )
    activation = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)

    actual = hybrid.apply_weight(activation, hybrid_weight)
    reference = mxfp6.gemm_from_float(
        activation,
        packed,
        out_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_fused_mlp_uses_packed_activation(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    weight = hybrid.HybridPackedWeight(
        torch.empty(0), torch.empty(0), rows=8, k=16
    )
    gate_up_state = hybrid.HybridState(
        route=hybrid.HybridRoute.ALL_ROWS,
        weights={},
        merged_weight=hybrid.HybridPackedWeight(
            torch.empty(0), torch.empty(0), rows=32, k=8
        ),
    )
    down_state = hybrid.HybridState(
        route=hybrid.HybridRoute.ALL_ROWS,
        weights={0: weight},
    )

    class _Linear:
        bias = None

        def __init__(self, state, output):
            self._mach_exl3_mxfp6 = state
            self.output = output

        def __call__(self, _x):
            return self.output, None

    gate_up = torch.ones((4, 32), dtype=torch.bfloat16)
    module = SimpleNamespace(
        gate_up_proj=_Linear(gate_up_state, gate_up),
        down_proj=_Linear(down_state, None),
        expert_gate=None,
    )
    module.down_proj.input_is_parallel = True
    module.down_proj.reduce_results = False
    module.down_proj.tp_size = 2
    packed = SimpleNamespace(k=16)
    seen = []
    monkeypatch.setattr(hybrid, "fused_silu_mxfp8", lambda value: packed)
    monkeypatch.setattr(
        hybrid,
        "apply_mxfp8_weight",
        lambda activation, selected: seen.append((activation, selected))
        or torch.full((4, 8), 3, dtype=torch.bfloat16),
    )

    output = fused_mlp._try_fused_forward(
        module, torch.ones((4, 8), dtype=torch.bfloat16)
    )

    assert torch.all(output == 3)
    assert seen == [(packed, weight)]


def test_fused_mlp_consumes_packed_input_without_requantizing(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)

    class _Packed:
        def __init__(self, rows, k):
            self.rows = rows
            self.k = k

    monkeypatch.setitem(sys.modules, "mxfp6", SimpleNamespace(MXFP8Tensor=_Packed))
    gate_up_weight = hybrid.HybridPackedWeight(
        torch.empty(0), torch.empty(0), rows=32, k=8
    )
    down_weight = hybrid.HybridPackedWeight(
        torch.empty(0), torch.empty(0), rows=8, k=16
    )

    class _Linear:
        bias = None

        def __init__(self, state):
            self._mach_exl3_mxfp6 = state

        def __call__(self, _x):
            raise AssertionError("packed input must bypass the BF16 linear boundary")

    module = SimpleNamespace(
        gate_up_proj=_Linear(
            hybrid.HybridState(
                route=hybrid.HybridRoute.ALL_ROWS,
                weights={},
                merged_weight=gate_up_weight,
            )
        ),
        down_proj=_Linear(
            hybrid.HybridState(
                route=hybrid.HybridRoute.ALL_ROWS,
                weights={0: down_weight},
            )
        ),
        expert_gate=None,
    )
    module.down_proj.input_is_parallel = True
    module.down_proj.reduce_results = False
    module.down_proj.tp_size = 2
    packed_input = _Packed(rows=4, k=8)
    packed_activation = _Packed(rows=4, k=16)
    calls = []

    def apply(activation, weight):
        calls.append((activation, weight))
        if weight is gate_up_weight:
            return torch.ones((4, 32), dtype=torch.bfloat16)
        return torch.full((4, 8), 3, dtype=torch.bfloat16)

    monkeypatch.setattr(hybrid, "apply_mxfp8_weight", apply)
    monkeypatch.setattr(hybrid, "fused_silu_mxfp8", lambda _value: packed_activation)

    output = fused_mlp._try_fused_forward(module, packed_input)

    assert torch.all(output == 3)
    assert calls == [
        (packed_input, gate_up_weight),
        (packed_activation, down_weight),
    ]


def test_fused_mlp_falls_back_outside_dense_hybrid_contract(monkeypatch) -> None:
    monkeypatch.setenv(hybrid.PROFILE_ENV, hybrid.QWEN38_27B_PROFILE)
    module = SimpleNamespace(
        gate_up_proj=SimpleNamespace(),
        down_proj=SimpleNamespace(),
        expert_gate=object(),
    )
    result = fused_mlp._try_fused_forward(
        module, torch.ones((4, 8), dtype=torch.float16)
    )
    assert result is fused_mlp._NOT_APPLICABLE


def test_fused_mlp_native_boundary_matches_runtime_reference() -> None:
    mxfp6 = pytest.importorskip("mxfp6")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    if not mxfp6.is_available():
        pytest.skip("mxfp6-sm120 is unavailable")

    rows, k, n = 4, 128, 128
    gate_up = torch.randn((rows, 2 * k), device="cuda", dtype=torch.bfloat16)
    actual_activation = hybrid.fused_silu_mxfp8(gate_up)
    reference_values, reference_logical_scales = (
        mxfp6.silu_and_mul_mxfp8_logical(gate_up)
    )

    torch.testing.assert_close(
        actual_activation.values,
        reference_values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        mxfp6.unpack_scales(actual_activation.scales, rows, k),
        reference_logical_scales,
        rtol=0,
        atol=0,
    )

    packed = mxfp6.quantize_mxfp6(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    weight = hybrid.HybridPackedWeight(
        values=packed.values,
        scales=packed.scales,
        rows=packed.rows,
        k=packed.k,
    )
    actual_output = hybrid.apply_mxfp8_weight(actual_activation, weight)
    reference_activation = mxfp6.MXFP8Tensor(
        reference_values.view(torch.uint8),
        mxfp6.pack_scales(reference_logical_scales),
        rows,
        k,
    )
    reference_output = mxfp6.gemm_w6a8(
        reference_activation,
        packed,
        out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)
