# SPDX-License-Identifier: Apache-2.0
"""Validated Qwen3.5-35B TP2 projection dispatch, installed before capture."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from vllm.logger import init_logger

from .dense import _import_mxfp6

logger = init_logger("vllm.mach.moe_projection")
_TABLE = Path(__file__).with_name("profile") / "qwen35_moe_projection.json"
_GEOMETRY = {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
}
_SHAPES = {(512, 2048), (2048, 256), (2048, 2048), (4608, 2048), (6144, 2048)}


def prepare(model, problems, dtype: torch.dtype) -> int:
    """Register only the measured native projection configurations.

    Native overrides are process/device/shape/dtype scoped, as are vLLM model
    workers. Expert grouped GEMMs use a separate dispatcher. Do not install
    these entries in a worker serving multiple models or retune after capture.
    """
    mode = os.environ.get("VLLM_MACH_MOE_PROJECTION_TUNING", "auto")
    if mode not in ("auto", "0", "1"):
        raise ValueError("VLLM_MACH_MOE_PROJECTION_TUNING must be auto, 0 or 1")
    if mode == "0" or dtype != torch.bfloat16 or not problems:
        return 0
    language_model = getattr(model, "language_model", model)
    config = getattr(language_model, "config", None)
    config = getattr(config, "text_config", config)
    if (getattr(config, "model_type", None) != "qwen3_5_moe_text"
            or any(getattr(config, key, None) != value
                   for key, value in _GEOMETRY.items())):
        return 0
    # Memory profiling invokes warmup outside set_current_vllm_config(). Read
    # the actual model's config rather than relying on a thread-local context.
    vllm_config = getattr(language_model, "vllm_config", None)
    if vllm_config is None:
        return 0
    parallel = vllm_config.parallel_config
    if (parallel.tensor_parallel_size != 2 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.enable_expert_parallel or parallel.enable_eplb
            or parallel.use_sequence_parallel_moe or parallel.use_ubatching
            or vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None):
        return 0
    if not _SHAPES.issubset({(n, k) for n, k, *_ in problems}):
        return 0
    device = problems[0][2].device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        return 0

    # The cache/tuner may otherwise replace the validated entries during warmup
    # or eager prefill. The Mach launcher already selects autotune=off.
    from mxfp6.autotune import is_autotune_enabled

    if is_autotune_enabled():
        if mode == "1":
            raise RuntimeError("MoE projection tuning requires MXFP6_AUTOTUNE=off")
        logger.info("Mach MoE projection table skipped: native autotuning is enabled")
        return 0

    extension = _import_mxfp6()
    extension.load_library()
    ops = torch.ops.mxfp6
    if not all(hasattr(ops, name) for name in ("w6a8_config_abi", "set_w6a8_config")):
        raise RuntimeError("MoE projection tuning requires an updated mxfp6-sm120 build")
    table = json.loads(_TABLE.read_text())
    anchor = torch.empty(0, device=device, dtype=torch.uint8)
    actual_abi = ops.w6a8_config_abi(anchor)
    if actual_abi != table["native_config_abi"]:
        raise RuntimeError(
            f"MoE projection config ABI mismatch: {actual_abi!r}; "
            f"expected {table['native_config_abi']!r}")
    for row in table["entries"]:
        if not ops.set_w6a8_config(
            anchor, row["m"], row["n"], row["k"],
            row["config"][0], row["config"][1], 0, dtype,
        ):
            raise RuntimeError(f"Could not register MoE projection config: {row}")
    count = len(table["entries"])
    logger.info("Mach installed %d Qwen3.5-35B TP2 projection configs (%s)", count, actual_abi)
    return count
