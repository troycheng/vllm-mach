# SPDX-License-Identifier: Apache-2.0
"""Retain the original softmax partitioning for the supported Mach profiles."""
import os

_NATIVE_REGISTERED = False


def enable_native():
    global _NATIVE_REGISTERED
    _NATIVE_REGISTERED = True


def preserve_old_softmax_segments(vllm_config):
    model = vllm_config.model_config
    if (str(model.dtype) not in ("bfloat16", "torch.bfloat16")
            or vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16")):
        return False
    profile = os.environ.get("VLLM_MACH_PROFILE")
    if profile:
        return profile == "qwen35-2b-mxfp8-champion-v1"
    if not _NATIVE_REGISTERED or model.quantization != "quark":
        return False
    text = model.hf_text_config
    if text.model_type not in ("qwen3_5_text", "qwen3_5_moe_text"):
        return False
    quant = model.model_arch_config.quantization_config
    if not isinstance(quant, dict):
        return False
    global_quant = quant.get("global_quant_config")
    if not isinstance(global_quant, dict):
        return False
    weight = global_quant.get("weight")
    activation = global_quant.get("input_tensors")
    if not isinstance(weight, dict) or not isinstance(activation, dict):
        return False
    return (weight.get("dtype") == "fp6_e3m2" and weight.get("group_size") == 32
            and weight.get("qscheme") == "per_group"
            and weight.get("scale_format") == "e8m0"
            and weight.get("is_dynamic") is False and weight.get("symmetric") is True
            and activation.get("dtype") == "fp8_e4m3"
            and activation.get("group_size") == 32
            and activation.get("qscheme") == "per_group"
            and activation.get("scale_format") == "e8m0"
            and activation.get("is_dynamic") is True
            and activation.get("symmetric") is True)
