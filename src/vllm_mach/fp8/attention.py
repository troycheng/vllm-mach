"""Metadata helpers for the narrow vLLM 0.29 mixed FA2 source backport.

The split algorithm comes from LiRunGuo's vLLM PR #58013. No runtime
function replacement or attention-kernel registration happens here.
"""

from __future__ import annotations

import os


def mixed_fa2_enabled() -> bool:
    return os.environ.get("VLLM_MACH_FP8_FA2") == "1"


def build_split_metadata(builder, common, metadata) -> None:
    """Attach a verified one-token decode prefix using existing CPU offsets."""
    metadata.num_split_decodes = 0
    metadata.prefill_query_start_loc = None
    cfg = builder.vllm_config
    if (
        not builder.mach_fp8_fa2
        or metadata.common_prefix_len
        or metadata.use_cascade
        or builder.dcp_world_size != 1
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.parallel_config.pipeline_parallel_size != 1
        or cfg.parallel_config.enable_dbo
        or cfg.speculative_config is not None
        or metadata.scheduler_metadata is not None
        or metadata.max_query_len <= 1
        or metadata.max_num_splits not in (0, 1)
        or metadata.mm_prefix_query_range_tensor is not None
        or metadata.rswa_prefix_lens is not None
        or metadata.causal is not True
    ):
        return
    starts = common.query_start_loc_cpu
    n = common.num_reqs
    if (
        starts.device.type != "cpu"
        or starts.numel() < n + 1
        or metadata.query_start_loc is not common.query_start_loc
    ):
        return
    offsets = starts[: n + 1].tolist()
    lengths = [b - a for a, b in zip(offsets, offsets[1:])]
    nd = next((i for i, length in enumerate(lengths) if length != 1), n)
    if (
        not 1 <= nd < n
        or not 1 <= n - nd <= 2
        or any(length <= 1 for length in lengths[nd:])
        or offsets[0] != 0
        or offsets[-1] != metadata.num_actual_tokens
        or max(lengths) != metadata.max_query_len
    ):
        return
    metadata.num_split_decodes = nd
    metadata.prefill_query_start_loc = metadata.query_start_loc[nd : n + 1] - nd


def mixed_fa2_eligible(impl, metadata, query, output, key_cache, value_cache) -> bool:
    """Keep unsupported geometry and capture on the original FA call."""
    import torch

    # AttentionType is a string enum in the pinned vLLM backend.
    attn_type = getattr(impl.attn_type, "name", impl.attn_type)
    nd = metadata.num_split_decodes
    starts = metadata.query_start_loc
    n = starts.numel() - 1
    device = query.device
    return (
        attn_type in ("decoder", "DECODER")
        and impl.vllm_flash_attn_version == 2
        and impl.dcp_world_size == 1
        and 1 <= nd < n
        and 1 <= n - nd <= 2
        and metadata.prefill_query_start_loc is not None
        and metadata.prefill_query_start_loc.shape == (n - nd + 1,)
        and metadata.prefill_query_start_loc.dtype == torch.int32
        and metadata.prefill_query_start_loc.device == device
        and metadata.max_num_splits in (0, 1)
        and metadata.max_query_len > 1
        and metadata.causal is True
        and not metadata.use_cascade
        and metadata.common_prefix_len == 0
        and metadata.scheduler_metadata is None
        and metadata.mm_prefix_query_range_tensor is None
        and metadata.rswa_prefix_lens is None
        and impl.alibi_slopes is None
        and impl.sinks is None
        and impl.logits_soft_cap == 0
        and (impl.sliding_window is None or tuple(impl.sliding_window) == (-1, -1))
        and query.shape == output.shape == (metadata.num_actual_tokens, 16, 256)
        and query.dtype == output.dtype == key_cache.dtype == value_cache.dtype
        == torch.bfloat16
        and key_cache.ndim == value_cache.ndim == 4
        and key_cache.shape[1:] == value_cache.shape[1:] == (528, 4, 256)
        and metadata.seq_lens.shape == (n,)
        and metadata.block_table.ndim == 2
        and metadata.block_table.shape[0] == n
        and starts.dtype == metadata.seq_lens.dtype == metadata.block_table.dtype
        == torch.int32
        and query.is_contiguous()
        and output.is_contiguous()
        and device == output.device == key_cache.device == value_cache.device
        == starts.device == metadata.seq_lens.device == metadata.block_table.device
        and not torch.cuda.is_current_stream_capturing()
    )
