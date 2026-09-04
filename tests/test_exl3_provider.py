from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from vllm.config import quantization as quantization_config_module
from vllm.model_executor.layers.quantization import get_quantization_config

from vllm_mach.exl3 import dense_adapter
from vllm_mach.exl3.dense_adapter import Exl3Config, Exl3LinearMethod
from vllm_mach.exl3.hadamard import hadamard_fold_weight_chunked
from vllm_mach.exl3.plugin import register


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


def test_registration_is_idempotent_and_does_not_patch_online_shorthands() -> None:
    before = quantization_config_module.ONLINE_QUANT_SHORTHAND_NAMES
    register()
    register()
    assert get_quantization_config("exl3") is Exl3Config
    assert quantization_config_module.ONLINE_QUANT_SHORTHAND_NAMES == before


def test_checkpoint_detection_and_metadata_validation() -> None:
    storage = _storage()
    assert Exl3Config.override_quantization_method(
        {"tensor_storage": storage}, None
    ) == "exl3"
    assert Exl3Config.override_quantization_method(
        {"tensor_storage": storage}, "awq"
    ) is None

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
