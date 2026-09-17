# SPDX-License-Identifier: Apache-2.0
"""Qwen35 TP2 native MoE reduction ownership for manual AR/Norm fusion."""

import torch
from vllm import envs
from vllm.config import CompilationMode
from vllm.platforms import current_platform


def use_moe_ar_norm(vllm_config) -> bool:
    config = vllm_config.model_config.hf_text_config
    parallel = vllm_config.parallel_config
    quant = vllm_config.quant_config
    scheme = getattr(quant, "quant_config", {}).get("global_quant_config", {})
    return (
        envs.VLLM_QWEN3_5_FUSED_AR_NORM
        and config.model_type == "qwen3_5_moe_text"
        and config.hidden_size == 2048
        and config.num_experts == 256
        and config.num_experts_per_tok == 8
        and config.moe_intermediate_size == 512
        and config.shared_expert_intermediate_size == 512
        and not getattr(config, "layer_scale", False)
        and vllm_config.model_config.dtype == torch.bfloat16
        and parallel.tensor_parallel_size == 2
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and not parallel.enable_expert_parallel
        and not parallel.enable_eplb
        and not parallel.use_sequence_parallel_moe
        and not parallel.use_ubatching
        and vllm_config.speculative_config is None
        and vllm_config.lora_config is None
        and vllm_config.compilation_config.mode
        in (CompilationMode.NONE, CompilationMode.VLLM_COMPILE)
        and quant is not None
        and quant.get_name() == "quark"
        and scheme.get("weight", {}).get("dtype") == "fp6_e3m2"
        and scheme.get("input_tensors", {}).get("dtype") == "fp8_e4m3"
        and current_platform.is_cuda()
        and current_platform.is_device_capability(120)
    )


def defer_moe_allreduce(block) -> None:
    """Return local routed + shared sums; the next fused norm owns the TP sum."""
    from .moe import Mxfp6Sm120MoERunner

    runner = block.experts
    if (
        not isinstance(runner, Mxfp6Sm120MoERunner)
        or block.replicate_shared_expert
        or runner.moe_config.is_sequence_parallel
        or runner.moe_config.tp_size != 2
        or runner.moe_config.ep_size != 1
        or runner.moe_config.dp_size != 1
    ):
        raise ValueError(
            "MoE AR/Norm fusion requires native MXFP6 TP2 local expert outputs"
        )
    runner.moe_config.skip_final_all_reduce = True
    # Authorize the existing small-batch schedules for this reduction owner.
    runner.routed_experts._mach_fused_ar_norm = True


def prepare_compiled_ar_norm(vllm_config) -> None:
    """Keep rank-specialized compiled model artifacts in separate namespaces."""
    if vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
        # vLLM's AOT Inductor directory is shared above rank-specific wrappers.
        # Manual model execution specializes embedding bounds and CUDA devices;
        # sharing this directory can load another rank's runnable on a cold run.
        vllm_config.additional_config["mach_compiled_ar_norm"] = {
            "version": 1,
            "rank": vllm_config.parallel_config.rank,
        }
