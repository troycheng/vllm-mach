from __future__ import annotations

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import quantization as quantization_config_module
from vllm.model_executor.kernels import linear as linear_kernels
from vllm.model_executor.kernels.linear.mxfp6.base import MxFp6LinearLayerConfig
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from vllm.platforms import PlatformEnum

from vllm_mach.exl3 import dense_adapter
from vllm_mach.exl3.dense_adapter import Exl3Config, Exl3LinearMethod
from vllm_mach.exl3.hadamard import hadamard_fold_weight_chunked
from vllm_mach.mxfp6 import Mxfp6Sm120LinearKernel, register_dense_kernel
from vllm_mach.mxfp6 import dense as mxfp6_dense
from vllm_mach.plugin import register


def _storage() -> dict[str, object]:
    return {
        "model.layers.0.self_attn.q_proj": {
            "quant_format": "exl3",
            "stored_tensors": {
                "q_proj.trellis": {},
                "q_proj.suh": {},
                "q_proj.svh": {},
                "q_proj.mcg": {},
            },
        }
    }


def _model_config() -> PretrainedConfig:
    config = PretrainedConfig(
        architectures=["Qwen3_5ForConditionalGeneration"],
        hidden_size=5120,
        tie_word_embeddings=False,
    )
    config.model_type = "qwen3_5"
    return config


def test_import_does_not_initialize_cuda() -> None:
    code = """
import torch
assert not torch.cuda.is_initialized()
import vllm_mach.exl3
assert not torch.cuda.is_initialized()
print(vllm_mach.exl3.Exl3Config().get_name())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "exl3"


def test_mxfp6_import_does_not_load_optional_module() -> None:
    code = """
import sys
import torch
assert "mxfp6" not in sys.modules
import vllm_mach.mxfp6
assert "mxfp6" not in sys.modules
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_registration_is_idempotent_and_does_not_patch_online_shorthands() -> None:
    before = quantization_config_module.ONLINE_QUANT_SHORTHAND_NAMES
    register()
    register()
    assert get_quantization_config("exl3") is Exl3Config
    assert quantization_config_module.ONLINE_QUANT_SHORTHAND_NAMES == before
    kernels = linear_kernels._POSSIBLE_MXFP6_KERNELS[PlatformEnum.CUDA]
    if importlib.util.find_spec("mxfp6") is None:
        assert Mxfp6Sm120LinearKernel not in kernels
    else:
        assert kernels[0] is Mxfp6Sm120LinearKernel
        assert kernels.count(Mxfp6Sm120LinearKernel) == 1


def test_mxfp6_kernel_accepts_only_native_w6a8_contract() -> None:
    native = MxFp6LinearLayerConfig(
        weight_quant_key=kMxfp6E3M2Static,
        activation_quant_key=kMxfp8Dynamic,
    )
    assert Mxfp6Sm120LinearKernel.can_implement(native) == (True, None)

    wrong_activation = MxFp6LinearLayerConfig(
        weight_quant_key=kMxfp6E3M2Static,
        activation_quant_key=None,
    )
    supported, reason = Mxfp6Sm120LinearKernel.can_implement(wrong_activation)
    assert not supported
    assert "MXFP8" in reason


def test_mxfp6_support_fails_closed_without_optional_runtime(monkeypatch) -> None:
    def missing_runtime():
        raise ModuleNotFoundError("mxfp6-sm120 is not installed")

    monkeypatch.setattr(mxfp6_dense, "_import_mxfp6", missing_runtime)
    assert not mxfp6_dense.is_mxfp6_sm120_available(90)
    assert not mxfp6_dense.is_mxfp6_sm120_available(120)


def test_mxfp6_registration_can_be_repeated_without_duplicate_kernel() -> None:
    kernels = linear_kernels._POSSIBLE_MXFP6_KERNELS[PlatformEnum.CUDA]
    original = list(kernels)
    try:
        if importlib.util.find_spec("mxfp6") is None:
            assert not register_dense_kernel()
            assert Mxfp6Sm120LinearKernel not in kernels
            return
        assert register_dense_kernel()
        assert register_dense_kernel()
        assert kernels[0] is Mxfp6Sm120LinearKernel
        assert kernels.count(Mxfp6Sm120LinearKernel) == 1
    finally:
        kernels[:] = original


def test_mxfp6_registration_does_not_import_optional_runtime(monkeypatch) -> None:
    kernels = linear_kernels._POSSIBLE_MXFP6_KERNELS[PlatformEnum.CUDA]
    original = list(kernels)

    def unexpected_import():
        raise AssertionError("registration imported mxfp6")

    monkeypatch.setattr("vllm_mach.mxfp6._optional_runtime_is_installed", lambda: True)
    monkeypatch.setattr(mxfp6_dense, "_import_mxfp6", unexpected_import)
    try:
        assert register_dense_kernel()
        assert kernels[0] is Mxfp6Sm120LinearKernel
    finally:
        kernels[:] = original


def test_mxfp6_activation_mapping_is_idempotent(monkeypatch) -> None:
    from vllm.model_executor.layers.quantization.quark.schemes import (
        quark_ocp_mx,
    )

    fake_mxfp6 = SimpleNamespace(
        **{name: object() for name in mxfp6_dense._REQUIRED_API}
    )
    monkeypatch.setattr(mxfp6_dense, "_import_mxfp6", lambda: fake_mxfp6)
    original = quark_ocp_mx._ACTIVATION_QUANT_KEY_MAP.get("mxfp8_e4m3")
    try:
        assert mxfp6_dense.register_vllm_mxfp8_activation()
        assert mxfp6_dense.register_vllm_mxfp8_activation()
        assert quark_ocp_mx._ACTIVATION_QUANT_KEY_MAP["mxfp8_e4m3"] == kMxfp8Dynamic
    finally:
        if original is None:
            quark_ocp_mx._ACTIVATION_QUANT_KEY_MAP.pop("mxfp8_e4m3", None)
        else:
            quark_ocp_mx._ACTIVATION_QUANT_KEY_MAP["mxfp8_e4m3"] = original


def test_mxfp6_activation_mapping_reaches_quark_scheme(monkeypatch) -> None:
    from vllm.model_executor.layers.quantization.quark.schemes import (
        quark_ocp_mx,
    )

    fake_mxfp6 = SimpleNamespace(
        **{name: object() for name in mxfp6_dense._REQUIRED_API}
    )
    monkeypatch.setattr(mxfp6_dense, "_import_mxfp6", lambda: fake_mxfp6)
    monkeypatch.setattr(
        quark_ocp_mx,
        "init_mxfp6_linear_kernel",
        lambda **kwargs: kwargs,
    )
    original = quark_ocp_mx._ACTIVATION_QUANT_KEY_MAP.get("mxfp8_e4m3")
    try:
        assert mxfp6_dense.register_vllm_mxfp8_activation()
        scheme = quark_ocp_mx.QuarkOCP_MX(
            {"dtype": "fp6_e3m2"},
            {"dtype": "fp8_e4m3", "is_dynamic": True},
        )
        assert scheme.input_dtype == "mxfp8_e4m3"
        assert scheme.activation_quant_key == kMxfp8Dynamic
    finally:
        if original is None:
            quark_ocp_mx._ACTIVATION_QUANT_KEY_MAP.pop("mxfp8_e4m3", None)
        else:
            quark_ocp_mx._ACTIVATION_QUANT_KEY_MAP["mxfp8_e4m3"] = original


def test_mxfp6_native_dense_matches_reference() -> None:
    mxfp6 = pytest.importorskip("mxfp6")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    if not mxfp6.is_available():
        pytest.skip("mxfp6-sm120 is unavailable")

    rows, input_features = 128, 128
    weights = torch.randn((rows, input_features), device="cuda", dtype=torch.float16)
    packed = mxfp6.quantize_mxfp6(weights)
    layer = SimpleNamespace(
        weight=packed.values.reshape(rows, -1).contiguous(),
        weight_scale=mxfp6.unpack_scales(
            packed.scales, rows, input_features
        ).contiguous(),
    )
    kernel = Mxfp6Sm120LinearKernel(
        MxFp6LinearLayerConfig(kMxfp6E3M2Static, kMxfp8Dynamic)
    )
    kernel.process_weights_after_loading(layer)
    x = torch.randn((4, input_features), device="cuda", dtype=torch.bfloat16)

    actual = kernel.apply_weights(layer, x)
    reference = mxfp6.gemm_from_float(
        x,
        mxfp6.PackedMXFP6Tensor(
            values=layer.weight,
            scales=layer.weight_scale,
            rows=rows,
            k=input_features,
        ),
        out_dtype=x.dtype,
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_mxfp6_rejects_invalid_packed_shape_before_scale_packing(
    monkeypatch,
) -> None:
    pack_called = False

    def pack_scales(scales):
        nonlocal pack_called
        pack_called = True
        return scales

    monkeypatch.setattr(
        mxfp6_dense,
        "_import_mxfp6",
        lambda: SimpleNamespace(pack_scales=pack_scales),
    )
    layer = SimpleNamespace(
        weight=torch.empty((7, 96), dtype=torch.uint8),
        weight_scale=torch.empty((7, 4), dtype=torch.uint8),
    )
    kernel = object.__new__(Mxfp6Sm120LinearKernel)

    with pytest.raises(ValueError, match="output features divisible by 8"):
        kernel.process_weights_after_loading(layer)
    assert not pack_called


def test_checkpoint_detection_and_metadata_validation() -> None:
    storage = _storage()
    assert (
        Exl3Config.override_quantization_method({"tensor_storage": storage}, None)
        == "exl3"
    )
    assert (
        Exl3Config.override_quantization_method({"tensor_storage": storage}, "awq")
        is None
    )

    config = Exl3Config.from_config({"bits": 5.5, "tensor_storage": storage})
    config.maybe_update_config(
        "unused-local-model",
        _model_config(),
    )
    assert config.get_name() == "exl3"
    assert config.bits == 5.5
    assert config.get_supported_act_dtypes() == [torch.float16, torch.bfloat16]
    assert config.get_min_capability() == 120


def test_unverified_model_layout_is_rejected() -> None:
    config = Exl3Config(tensor_storage=_storage())
    other = PretrainedConfig(
        architectures=["OtherForCausalLM"],
        hidden_size=4096,
        tie_word_embeddings=False,
    )
    other.model_type = "other"
    try:
        config.maybe_update_config("unused-local-model", other)
    except ValueError as exc:
        assert "refusing unverified configuration" in str(exc)
    else:
        raise AssertionError("an unverified model layout was accepted")


def test_invalid_metadata_is_rejected_before_loading() -> None:
    config = Exl3Config(
        tensor_storage={
            "model.layers.0.self_attn.q_proj": {
                "quant_format": "exl3",
                "stored_tensors": {"q_proj.trellis": {}},
            }
        }
    )
    try:
        config.maybe_update_config("unused-local-model")
    except ValueError as exc:
        assert "missing suh|su,svh|sv" in str(exc)
    else:
        raise AssertionError("invalid EXL3 metadata was accepted")


def test_bundled_hadamard_fold_identity_scales() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(128, 128, generator=generator, dtype=torch.float32)
    original = weight.clone()
    scale = torch.ones(128, dtype=torch.float32)
    folded = hadamard_fold_weight_chunked(weight, scale, scale)
    # An orthogonal Hadamard on both axes preserves the Frobenius norm.
    torch.testing.assert_close(
        torch.linalg.vector_norm(folded),
        torch.linalg.vector_norm(original),
        rtol=2e-6,
        atol=2e-6,
    )


def test_serialized_apply_one_has_a_defined_fallback(monkeypatch) -> None:
    trellis = torch.empty((8, 8, 80), dtype=torch.int16)
    scales = torch.ones(128, dtype=torch.float16)
    empty = SimpleNamespace(exl3_tensors={})
    layer = SimpleNamespace(
        trellis=SimpleNamespace(exl3_tensors={None: trellis}),
        suh=SimpleNamespace(exl3_tensors={None: scales}),
        svh=SimpleNamespace(exl3_tensors={None: scales}),
        mcg=empty,
        mul1=empty,
        exl3_output_partition_sizes=[128],
        exl3_shard_ids=[None],
    )
    monkeypatch.setattr(
        dense_adapter,
        "_b12x_trellis_k6_supported",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        dense_adapter,
        "_exl3_gemm",
        lambda x, *args: torch.zeros((x.shape[0], 128), dtype=torch.float16),
    )
    output = Exl3LinearMethod._apply_one(
        layer,
        torch.zeros((4, 128), dtype=torch.float16),
        None,
    )
    assert output.shape == (4, 128)
