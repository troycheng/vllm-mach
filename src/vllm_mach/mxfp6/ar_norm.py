# SPDX-License-Identifier: Apache-2.0
"""Dense TP2 AR/GemmaRMSNorm/MXFP8 with an explicit producer-consumer handoff.

The native kernel preserves the existing BF16 norm rounding before quantization.
Only the validated eager/FULL-graph decode boundary is replaced. There is no
activation cache: each invocation owns its outputs and uses vLLM's AR workspace.
"""

from __future__ import annotations

import os
import types
from collections import Counter
from importlib import metadata, util

import torch

ABI = "ar-norm-mxfp8-v1"
_ROWS = (2, 4, 8, 16, 24, 32)
_STATS = Counter()
_loaded = False


def stats():
    """Host dispatch/capture counts, not CUDA Graph replay counts."""
    return dict(_STATS)


def load_library():
    """Load the installed wheel; never build code during service startup."""
    global _loaded
    if _loaded:
        return
    for package, expected in (
        ("flashinfer-python", "0.6.18"),
        ("mxfp6-sm120", "0.2.1"),
        ("vllm-mach-ar-norm", "0.1.0a1"),
    ):
        try:
            version = metadata.version(package).split("+", 1)[0]
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"AR/MXFP8 requires {package}=={expected}") from exc
        if version != expected:
            raise RuntimeError(f"AR/MXFP8 requires {package}=={expected}; found {version}")
    spec = util.find_spec("mach_ar_norm_ext")
    if spec is None or spec.origin is None:
        raise RuntimeError("Build and install native/ar_norm before enabling AR/MXFP8")
    torch.ops.load_library(spec.origin)
    ops = torch.ops.mach_norm_quant
    if not hasattr(ops, "abi") or ops.abi() != ABI or not hasattr(ops, "run"):
        raise RuntimeError("AR/MXFP8 native ABI mismatch; rebuild native/ar_norm")
    from .dense import _import_mxfp6

    _import_mxfp6().load_library()
    if not hasattr(torch.ops.mxfp6, "gemm_w6a8_pdl"):
        raise RuntimeError("AR/MXFP8 requires mxfp6 gemm_w6a8_pdl; rebuild mxfp6-sm120")
    _loaded = True


def _eligible(layer):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    from .dense import Mxfp6Sm120LinearKernel

    # use_fused_ar_gemma_norm already requires BF16/SM120/TP2/PP1, no LoRA,
    # no speculation and CompilationMode.NONE for the Dense model.
    if (type(layer) is not Qwen3_5DecoderLayer
            or not layer.use_fused_ar_gemma_norm or layer.layer_scale
            or not getattr(layer.mlp, "_mach_swiglu_prepared", False)):
        return False
    norms = (layer.input_layernorm, layer.post_attention_layernorm)
    if not all(type(norm) is GemmaRMSNorm and norm.weight.shape == (5120,)
               and norm.weight.dtype == torch.bfloat16 and norm.weight.is_cuda
               and norm.weight.is_contiguous() for norm in norms):
        return False
    if torch.cuda.get_device_capability(norms[0].weight.device) != (12, 0):
        return False
    if layer.layer_type == "full_attention":
        return True
    if layer.layer_type != "linear_attention":
        return False
    attention = layer.linear_attn
    projection = attention.in_proj_qkvz
    return (getattr(attention, "_mach_gdn_prepared", False)
            and projection.tp_size == 2 and projection.bias is None
            and tuple(projection.weight.shape) == (8192, 3840)
            and isinstance(getattr(getattr(projection, "scheme", None),
                                   "ocp_mx_linear", None), Mxfp6Sm120LinearKernel))


def decode_supported(x, prefix=None):
    if (x.ndim != 2 or x.shape[1] != 5120 or x.shape[0] not in _ROWS
            or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous()
            or torch.compiler.is_compiling()):
        return False
    from vllm.forward_context import get_forward_context

    raw = get_forward_context().attn_metadata
    if not isinstance(raw, dict):
        return False
    md = raw.get(prefix) if prefix else next(
        (v for v in raw.values() if hasattr(v, "num_spec_decodes")), None)
    return bool(md is not None and getattr(md, "num_prefills", None) == 0
                and getattr(md, "num_spec_decodes", None) == 0
                and getattr(md, "spec_sequence_masks", True) is None
                and getattr(md, "num_decodes", 0) > 0
                and getattr(md, "num_actual_tokens", None) == x.shape[0])


def _workspace(x):
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tp_group
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        _can_use_flashinfer, get_fi_ar_workspace,
    )

    ok, max_tokens = _can_use_flashinfer(x, 2)
    if not ok:
        return None
    rank = get_tensor_model_parallel_rank()
    workspace = get_fi_ar_workspace(world_size=2, rank=rank, max_token_num=max_tokens,
                                   hidden_dim=5120, dtype=x.dtype,
                                   group=get_tp_group().cpu_group)
    if workspace is None or workspace.backend != "trtllm":
        return None
    return workspace.workspace_tensor, rank


def normalize(x, residual, norm, workspace):
    normalized = torch.empty_like(x)
    codes = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
    scales = torch.empty(128 * 160, device=x.device, dtype=torch.uint8)
    ws, rank = workspace
    # As in vLLM's original boundary, x becomes the new residual. The incoming
    # residual is read-only; the BF16 normalized output is also needed by BA.
    torch.ops.mach_norm_quant.run(x, residual, norm.weight, ws,
                                 normalized, x, codes, scales,
                                 rank, norm.variance_epsilon, True)
    return normalized, x, codes, scales


def projected(codes, scales, layer):
    return torch.ops.mxfp6.gemm_w6a8_pdl(
        codes, layer.weight, scales, layer.weight_scale,
        codes.shape[0], layer.weight.shape[0], 5120, 1.0, torch.bfloat16)


def _forward(layer, hidden_states, residual, positions=None, *, original, **kwargs):
    prefix = layer.linear_attn.prefix if layer.layer_type == "linear_attention" else None
    supported = decode_supported(hidden_states, prefix)
    if residual is not None:
        supported = supported and (residual.shape == hidden_states.shape
                                  and residual.dtype == hidden_states.dtype
                                  and residual.device == hidden_states.device
                                  and residual.is_contiguous())
    workspace = _workspace(hidden_states) if supported else None
    if workspace is None:
        return original(hidden_states=hidden_states, residual=residual,
                        positions=positions, **kwargs)
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        fused_allreduce_gemma_rms_norm,
    )

    rows = hidden_states.shape[0]
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = (layer.linear_attn(hidden_states=hidden_states)
                         if layer.layer_type == "linear_attention" else
                         layer.self_attn(hidden_states=hidden_states, positions=positions))
    elif layer.layer_type == "linear_attention":
        normalized, residual, q, s = normalize(hidden_states, residual,
                                               layer.input_layernorm, workspace)
        # QKV stays inside GDN, after the auxiliary BA launch. Passing tensors
        # explicitly avoids activation-address caches and preserves stream order.
        hidden_states = layer.linear_attn._forward_method(normalized, quantized=(q, s))
        _STATS[f"input:M{rows}"] += 1
    else:
        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, layer.input_layernorm)
        hidden_states = layer.self_attn(hidden_states=hidden_states, positions=positions)
    _, residual, q, s = normalize(hidden_states, residual,
                                layer.post_attention_layernorm, workspace)
    gate_up = projected(q, s, layer.mlp.gate_up_proj)
    output = torch.ops.vllm.mach_swiglu_mxfp6_down(
        gate_up, layer.mlp.down_proj.weight, layer.mlp.down_proj.weight_scale)
    _STATS[f"post:M{rows}"] += 1
    return output, residual


def prepare(model):
    """Prepare eligible Dense layers after GDN/MLP producers, before capture."""
    mode = os.environ.get("VLLM_MACH_FUSED_AR_QUANT", "auto")
    if mode not in ("auto", "0", "1"):
        raise ValueError("VLLM_MACH_FUSED_AR_QUANT must be auto, 0 or 1")
    if mode == "0":
        return 0
    layers = [layer for layer in model.modules()
              if not getattr(layer, "_mach_ar_quant_prepared", False) and _eligible(layer)]
    if not layers:
        return 0
    load_library()
    from functools import partial

    for layer in layers:
        layer.forward = types.MethodType(partial(_forward, original=layer.forward), layer)
        layer._mach_ar_quant_prepared = True
    _STATS["prepared_layers"] += len(layers)
    from vllm.logger import init_logger

    init_logger("vllm.mach.ar_norm").info(
        "Mach prepared %d Dense AR/GemmaRMSNorm/MXFP8 layers (M2/4/8/16/24/32)", len(layers))
    return len(layers)
