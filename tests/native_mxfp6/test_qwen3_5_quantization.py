# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_manual_ar_norm_rejects_unsupported_execution_modes(monkeypatch):
    """Delayed row reductions must not escape into PP, LoRA, MoE, or Eagle."""
    from types import SimpleNamespace as NS

    import torch
    from vllm.config import CompilationMode
    from vllm.model_executor.models import qwen3_5 as model

    monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "1")
    cfg = NS(
        model_config=NS(
            dtype=torch.bfloat16,
            hf_text_config=NS(
                model_type="qwen3_5_text", hidden_size=5120, layer_scale=False
            ),
        ),
        parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1),
        compilation_config=NS(mode=CompilationMode.NONE),
        quant_config=NS(get_name=lambda: "quark"),
        speculative_config=None,
        lora_config=None,
    )
    with patch.object(model, "current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.is_device_capability.return_value = True
        assert model._use_mxfp6_ar_norm(cfg)
        for obj, attr, value in (
            (cfg, "speculative_config", object()),
            (cfg, "lora_config", object()),
            (cfg.parallel_config, "pipeline_parallel_size", 2),
            (cfg.model_config.hf_text_config, "model_type", "qwen3_5_moe_text"),
            (cfg.model_config.hf_text_config, "layer_scale", True),
            (cfg.compilation_config, "mode", CompilationMode.VLLM_COMPILE),
        ):
            old = getattr(obj, attr)
            setattr(obj, attr, value)
            assert not model._use_mxfp6_ar_norm(cfg)
            setattr(obj, attr, old)
        monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "0")
        assert not model._use_mxfp6_ar_norm(cfg)


def test_fp16_gdn_admission_keeps_speculative_and_other_geometries_out(monkeypatch):
    """FP16 storage must not reach the BF16/FP32-only fused MTP kernel."""
    from types import SimpleNamespace as NS

    import torch
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    layer = NS(
        get_state_dtype=lambda: (torch.bfloat16, torch.float16),
        tp_size=2,
        hidden_size=5120,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        enable_packed_recurrent_decode=True,
        gqa_interleaved_layout=False,
        norm=NS(activation="silu"),
    )
    config = NS(model_config=NS(dtype=torch.bfloat16), speculative_config=None)
    guard = gdn.QwenGatedDeltaNetAttention._fused_gdn_decode_unsupported_reason
    monkeypatch.setattr(
        torch.ops._C, "fused_gdn_decode_post_conv_mtp", object(), raising=False
    )
    with patch.object(gdn, "current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.is_device_capability.return_value = True
        platform.has_device_capability.return_value = True
        monkeypatch.setenv("VLLM_QWEN3_5_FP16_SSM", "0")
        assert guard(layer, config) is not None
        monkeypatch.setenv("VLLM_QWEN3_5_FP16_SSM", "1")
        assert guard(layer, config) is None
        assert torch.float16 not in gdn.FUSED_GDN_STATE_DTYPES
        for obj, attr, value in (
            (config, "speculative_config", object()),
            (layer, "tp_size", 1),
            (layer, "hidden_size", 4096),
            (layer, "gqa_interleaved_layout", True),
            (layer, "enable_packed_recurrent_decode", False),
            (layer, "get_state_dtype", lambda: (torch.float16, torch.float16)),
        ):
            old = getattr(obj, attr)
            setattr(obj, attr, value)
            assert guard(layer, config) is not None
            setattr(obj, attr, old)
        platform.is_device_capability.return_value = False
        assert guard(layer, config) is not None
