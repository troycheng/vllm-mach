"""Public profile selection and source-install isolation without CUDA."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from vllm_mach.fp8 import profile, serve


def test_launch_keeps_full_capacity_precision_and_independent_features():
    with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
        first = serve.configure_environment(temporary, features=("ordered", "fa2"))
        assert profile.selected_features() == ("ordered", "fa2")
        assert first["VLLM_MACH_FP8_N64"] == first["VLLM_MACH_FP8_SILU"] == "0"
        assert first["VLLM_MACH_FP8_COMPILE_MODE"] == "production"
        assert first["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        second = serve.configure_environment(temporary, features=("n64",))
        assert profile.selected_features() == ("n64",)
        assert first["VLLM_CACHE_ROOT"] != second["VLLM_CACHE_ROOT"]
        assert Path(first["VLLM_CACHE_ROOT"]).parent.is_dir()
    args = serve.build_argv("/models/block", tokenizer="/models/tokenizer")
    expected = {"--max-num-seqs": "128", "--max-num-batched-tokens": "2048",
                "--max-model-len": "8192", "--kv-cache-memory": str(19 * 2**30),
                "--mamba-ssm-cache-dtype": "float32", "--dtype": "bfloat16",
                "--linear-backend": "cutlass", "--attention-backend": "FLASH_ATTN",
                "--tokenizer": "/models/tokenizer"}
    assert all(args[args.index(key) + 1] == value for key, value in expected.items())


@pytest.mark.parametrize("features", [(), ("n64", "n64"), ("unknown",)])
def test_bad_feature_selection_has_no_environment_or_directory_side_effect(features):
    with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
        target = Path(temporary) / "absent"
        with pytest.raises(ValueError):
            serve.configure_environment(target, features=features)
        assert not target.exists() and not os.environ


def test_profile_refuses_combined_lossy_flags_or_dependency_drift():
    settings = {"VLLM_MACH_PROFILE": profile.NAME, "VLLM_MACH_FP8_N64": "1"}
    with patch.dict(os.environ, settings, clear=True):
        with patch.object(profile, "version", side_effect=profile.DEPENDENCIES.__getitem__):
            assert profile.check_environment() == profile.DEPENDENCIES
            os.environ["VLLM_HYBRID_NVFP4_LM_HEAD"] = "1"
            with pytest.raises(RuntimeError, match="cannot be combined"):
                profile.check_environment()
        del os.environ["VLLM_HYBRID_NVFP4_LM_HEAD"]
        with patch.object(profile, "version", return_value="0.30.0"):
            with pytest.raises(RuntimeError, match="requires"):
                profile.check_environment()


def test_nested_weight_offload_configs_require_a_different_profile():
    from vllm_mach.fp8.worker import validate_worker_config

    cfg = NS(
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           decode_context_parallel_size=1, enable_dbo=False),
        speculative_config=None,
        model_config=NS(dtype="torch.bfloat16", quantization="compressed-tensors",
            hf_text_config=NS(model_type="qwen3_5_text", hidden_size=2560,
                intermediate_size=9216, num_hidden_layers=32, num_attention_heads=16,
                num_key_value_heads=4, head_dim=256)),
        cache_config=NS(cache_dtype="auto", mamba_ssm_cache_dtype="float32"),
        offload_config=NS(uva=NS(cpu_offload_gb=0), prefetch=NS(offload_group_size=0)),
    )
    worker = NS(vllm_config=cfg)
    validate_worker_config(worker)
    for section, field in (("uva", "cpu_offload_gb"), ("prefetch", "offload_group_size")):
        setattr(getattr(cfg.offload_config, section), field, 1)
        with pytest.raises(RuntimeError, match="offload"):
            validate_worker_config(worker)
        setattr(getattr(cfg.offload_config, section), field, 0)
