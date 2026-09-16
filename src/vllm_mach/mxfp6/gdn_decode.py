# SPDX-License-Identifier: Apache-2.0
"""Native Qwen 27B TP2 GDN decode paths, prepared before profiling.

Persistent uses FP32 or FP16 SSM at physical M1/2/4/8. BA overlap uses M16/24/32
with FP32 or FP16 SSM. Prefill, speculative, mixed and unsupported calls
retain the original vLLM method. Shared persistent scratch assumes vLLM's
serialized worker execution, including CUDA Graph replay.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import partial

import torch

PERSISTENT_ROWS = (1, 2, 4, 8)
OVERLAP_ROWS = (16, 24, 32)
_STATS = Counter()


def enabled(name: str) -> bool:
    return os.environ.get("VLLM_MACH_GDN_" + name) == "1"


def stats() -> dict:
    """Host dispatch/capture counts, not counts of CUDA Graph replays."""
    return dict(_STATS)


def select_path(rows, state_dtype, metadata, *, persistent, overlap):
    if (
        metadata is None
        or metadata.spec_sequence_masks is not None
        or metadata.num_spec_decodes != 0
        or metadata.num_prefills != 0
        or metadata.num_decodes <= 0
        or metadata.num_actual_tokens != rows
        or metadata.non_spec_state_indices_tensor is None
    ):
        return None
    if (
        persistent
        and rows in PERSISTENT_ROWS
        and state_dtype in (torch.float32, torch.float16)
    ):
        return "persistent"
    if (
        overlap
        and rows in OVERLAP_ROWS
        and state_dtype in (torch.float16, torch.float32)
    ):
        return "overlap"
    return None


def _eligible_layer(layer):
    return (
        type(layer).__name__ == "QwenGatedDeltaNetAttention"
        and layer.tp_size == 2
        and layer.num_k_heads == 16
        and layer.num_v_heads == 48
        and layer.head_k_dim == layer.head_v_dim == 128
        and not layer.gqa_interleaved_layout
        and not layer.disable_tp_for_ba_proj
        and layer.enable_fused_gdn_decode
        and layer.enable_packed_recurrent_decode
        and layer.activation in ("silu", "swish")
        and tuple(layer.in_proj_ba.weight.shape) == (48, 5120)
        and layer.in_proj_ba.weight.dtype == torch.bfloat16
        and getattr(layer.in_proj_ba, "bias", None) is None
        and tuple(layer.conv1d.weight.shape) == (5120, 1, 4)
        and layer.conv1d.weight.dtype == torch.bfloat16
        and (
            getattr(layer.conv1d, "bias", None) is None
            or layer.conv1d.bias.dtype == torch.bfloat16
        )
        and layer.A_log.dtype == torch.float32
        and layer.dt_bias.dtype == torch.bfloat16
        and layer.norm.weight.dtype in (torch.bfloat16, torch.float32)
    )


def _forward(layer, original, persistent, aux, hidden_states):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    rows = hidden_states.shape[0]
    if (
        hidden_states.shape != (rows, 5120)
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
    ):
        return original(hidden_states)
    raw = get_forward_context().attn_metadata
    md = raw.get(layer.prefix) if isinstance(raw, dict) else None
    if md is None or not layer.kv_cache:
        return original(hidden_states)
    conv, state = layer.kv_cache
    path = select_path(
        rows, state.dtype, md, persistent=persistent, overlap=aux is not None
    )
    if path is None or conv.dtype != torch.bfloat16:
        return original(hidden_states)
    conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
    indices = md.non_spec_state_indices_tensor[:rows]
    # Shape/stride checks stay on the host; index values remain on device.
    if (
        conv.shape[1:] != (5120, 3)
        or state.shape[1:] != (24, 128, 128)
        or state.stride(-1) != 1
        or state.stride(-2) != 128
        or state.stride(-3) != 128 * 128
        or indices.ndim != 1
        or indices.shape[0] != rows
        or indices.dtype != torch.int32
        or indices.stride(0) != 1
    ):
        return original(hidden_states)
    core = torch.zeros(
        (rows, 24, 128), device=hidden_states.device, dtype=torch.bfloat16
    )
    if path == "persistent":
        from .gdn import persistent as kernel

        mixed, _ = layer.in_proj_qkvz(hidden_states)
        qkv, z = mixed.split([5120, 3072], -1)
        kernel.execute(
            hidden_states,
            layer._mach_gdn_ba,
            qkv,
            layer.conv1d.weight.view(5120, 4),
            layer._mach_gdn_bias,
            conv,
            layer.A_log,
            layer.dt_bias,
            128**-0.5,
            state,
            indices,
            core.unsqueeze(1),
        )
    else:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )
        from vllm.third_party.flash_linear_attention.ops import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        main = torch.cuda.current_stream(hidden_states.device)
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            ba, _ = layer.in_proj_ba(hidden_states)
            b, a = layer.split_ba(ba)
            b, a = b.contiguous(), a.contiguous()
        mixed, _ = layer.in_proj_qkvz(hidden_states)
        qkv, z = mixed.split([5120, 3072], -1)
        qkv = causal_conv1d_update(
            qkv,
            conv,
            layer.conv1d.weight.view(5120, 4),
            layer.conv1d.bias,
            layer.activation,
            conv_state_indices=indices,
            validate_data=False,
        )
        main.wait_stream(aux)
        ba.record_stream(main)
        b.record_stream(main)
        a.record_stream(main)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=qkv,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=128**-0.5,
            initial_state=state,
            out=core.unsqueeze(1),
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
    from .gdn_output import project as output_projection

    output = output_projection(layer, core, z)
    key = f"{path}_m{rows}"
    if not _STATS[key]:
        from vllm.logger import init_logger

        init_logger("vllm.mach.gdn").info(
            "Mach GDN selected %s at physical M%d", path, rows
        )
    _STATS[key] += 1
    return output


@torch.inference_mode()
def prepare(model):
    """Install once on loaded native MXFP6 layers, compile/warm before capture."""
    persistent, overlap = enabled("PERSISTENT"), enabled("BA_OVERLAP")
    if not (persistent or overlap):
        return
    from .dense import Mxfp6Sm120LinearKernel
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.logger import init_logger

    layers = []
    for layer in model.modules():
        if not _eligible_layer(layer) or hasattr(layer, "_mach_gdn_prepared"):
            continue
        kernel = getattr(
            getattr(layer.in_proj_qkvz, "scheme", None), "ocp_mx_linear", None
        )
        if not isinstance(kernel, Mxfp6Sm120LinearKernel):
            continue
        if layer.get_state_dtype()[1] not in (torch.float16, torch.float32):
            continue
        device = layer.in_proj_ba.weight.device
        if not device.type == "cuda" or torch.cuda.get_device_capability(device) != (
            12,
            0,
        ):
            continue
        layers.append(layer)
    if not layers:
        if any(hasattr(m, "_mach_gdn_prepared") for m in model.modules()):
            return
        init_logger("vllm.mach.gdn").warning(
            "Mach GDN requested but no eligible native MXFP6 layers found"
        )
        return
    device = layers[0].in_proj_ba.weight.device
    aux = torch.cuda.Stream(device=device) if overlap else None
    if persistent:
        from .gdn import persistent as kernel
    from .gdn_output import prepare as prepare_output

    for layer in layers:
        prepare_output(layer)
        if persistent:
            layer._mach_gdn_ba = layer.in_proj_ba.weight.T.contiguous()
            layer._mach_gdn_bias = layer.conv1d.bias
            if layer._mach_gdn_bias is None:
                layer._mach_gdn_bias = torch.zeros(
                    5120, device=device, dtype=torch.bfloat16
                )
        if aux is not None:
            aux.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(aux):
                for rows in OVERLAP_ROWS:
                    layer.in_proj_ba(
                        torch.zeros((rows, 5120), device=device, dtype=torch.bfloat16)
                    )
            torch.cuda.current_stream(device).wait_stream(aux)
    if persistent:
        layer = layers[0]
        # Warmup runs outside vLLM's global config context. Read the resolved
        # cache dtype from the loaded layer, just as its state allocator does.
        state_dtype = layer.get_state_dtype()[1]
        for rows in PERSISTENT_ROWS:
            x = torch.zeros((rows, 5120), device=device, dtype=torch.bfloat16)
            conv = torch.zeros(
                (rows + 1, 3, 5120), device=device, dtype=torch.bfloat16
            ).transpose(1, 2)
            if is_conv_state_dim_first():
                conv = conv.contiguous()
            state = torch.zeros(
                (rows + 1, 24, 128, 128), device=device, dtype=state_dtype
            )
            indices = torch.arange(1, rows + 1, device=device, dtype=torch.int32)
            kernel.execute(
                x,
                layer._mach_gdn_ba,
                x,
                layer.conv1d.weight.view(5120, 4),
                layer._mach_gdn_bias,
                conv,
                layer.A_log,
                layer.dt_bias,
                128**-0.5,
                state,
                indices,
            )
    torch.cuda.synchronize(device)
    for layer in layers:
        layer._forward_method = partial(
            _forward, layer, layer._forward_method, persistent, aux
        )
        layer._mach_gdn_prepared = True
    _STATS["prepared_layers"] += len(layers)
    _STATS["fused_output_layers"] += sum(bool(getattr(l, "_mach_gdn_output_fused", False)) for l in layers)
    init_logger("vllm.mach.gdn").info(
        "Mach GDN prepared %d layers: persistent=%s rows=%s, BA overlap=%s rows=%s",
        len(layers),
        persistent,
        PERSISTENT_ROWS,
        overlap,
        OVERLAP_ROWS,
    )
