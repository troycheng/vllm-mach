# SPDX-License-Identifier: Apache-2.0
"""Selected M32 NVFP4 dual-A4 gate/up with rank64 compensation.

All non-M32 rows retain the MXFP6 route. Assets and persistent buffers are
loaded before graph capture; the two GEMM branches retain their fork/join.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from . import mxfp6_hybrid
from .rank64_assets import load_entry, load_manifest, tensor_sha256

ENV = "VLLM_MACH_RANK64_BUNDLE"
ATTRIBUTE = "_vllm_mach_rank64"
_STREAMS: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _bundle(path: str) -> dict:
    return load_manifest(Path(path))


def _scale_index(device: torch.device) -> torch.Tensor:
    # Invert the actual FlashInfer interleave, rather than assuming its layout.
    from flashinfer.quantization.fp4_quantization import block_scale_interleave
    labels = torch.arange(128 * 320, device=device, dtype=torch.int64).view(128, 320)
    logical = torch.zeros_like(labels)
    for shift in (0, 8, 16):
        logical |= block_scale_interleave(((labels >> shift) & 255).to(torch.uint8)).view(128, 320).long() << shift
    physical = torch.empty_like(labels).flatten()
    physical[logical.flatten()] = torch.arange(128 * 320, device=device)
    return physical.view(128, 320)[:64].contiguous().int()


class Rank64GateUp:
    def __init__(self, tensors: dict, entry: dict, device: torch.device):
        from . import rank64_prepare
        import flashinfer
        self._gemm = flashinfer.gemm.mm_fp4
        self._prepare = rank64_prepare.prepare_into
        self.packed, self.scales, self.global_scale, self.a, self.b = (
            tensors[name].to(device=device) for name in
            ("packed", "scales", "global_scale", "aware64_a", "aware64_b")
        )
        self.a_global_scale = torch.full((1,), entry["activation_global_scale"], dtype=torch.float32, device=device)
        self.alpha = torch.reciprocal(self.a_global_scale * self.global_scale)
        self.index = _scale_index(device)
        self.buffers = rank64_prepare.make_buffers(device)
        self.z = torch.empty((32, 64), dtype=torch.bfloat16, device=device)
        key = str(device)
        if key not in _STREAMS:
            _STREAMS[key] = torch.cuda.Stream(device=device)
        self.aux_stream = _STREAMS[key]
        self.entry = entry
        self.graph_rows: set[int] = set()
        self.graph_input: torch.Tensor | None = None
        self.norm_verified = False

    def verify_norm(self, norm: Any) -> None:
        if self.norm_verified:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("rank64 requires one eager forward to verify its RMSNorm before graph capture")
        weight = norm.weight.detach()
        if tuple(weight.shape) != (5120,) or float(norm.variance_epsilon) != 1e-6:
            raise ValueError("rank64 RMSNorm geometry/epsilon mismatch")
        # vLLM stores the raw Gemma RMSNorm weight; the +1 is applied by the op.
        if tensor_sha256(weight.to(torch.bfloat16)) != self.entry["norm_bf16_sha256"]:
            raise ValueError("rank64 static scale was prepared for another RMSNorm weight")
        self.norm_verified = True

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        from .rank64_merge import up_merge
        if not isinstance(x, torch.Tensor) or tuple(x.shape) != (32, 5120) or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != self.packed.device:
            raise ValueError("rank64 requires contiguous BF16 [32,5120] on its weight device")
        current = torch.cuda.current_stream()
        self.aux_stream.wait_stream(current)
        with torch.cuda.stream(self.aux_stream):
            torch.mm(x, self.a, out=self.z)
        prepared = self._prepare(x, self.a_global_scale, self.index, self.buffers)
        terms = self._gemm(
            prepared["a_packed"], self.packed.T, prepared["a_scales"], self.scales.T,
            self.alpha, out_dtype=torch.bfloat16, block_size=16,
            use_8x4_sf_layout=False, backend="b12x", use_nvfp4=True, enable_pdl=False,
        )
        current.wait_stream(self.aux_stream)
        result = up_merge(terms, self.z, self.b)
        if torch.cuda.is_current_stream_capturing():
            self.graph_rows.add(32)
            self.graph_input = x
        return result


def attach_layer(layer: Any) -> bool:
    path = os.environ.get(ENV, "")
    if not path:
        return False
    data = _bundle(path)
    match = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.mlp\.gate_up_proj", str(getattr(layer, "prefix", "")))
    if match is None or int(match[1]) not in data["mask"]:
        return False
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != 2:
        raise ValueError("rank64 bundle requires TP2")
    state = mxfp6_hybrid.state_for_rows(layer, 32)
    if state is None or state.merged_weight is None:
        raise ValueError("rank64 requires the MXFP6 checkpoint gate/up route")
    if os.environ.get("VLLM_MACH_EXL3_MXFP6_FUSED_MLP", "1") != "1" or os.environ.get("VLLM_MACH_EXL3_MXFP6_FUSED_AR_NORM_MXFP8", "0") != "1":
        raise ValueError("rank64 requires the fused MLP and AR/RMSNorm runtime profile")
    if hasattr(layer, ATTRIBUTE):
        raise ValueError("rank64 gate/up is already attached")
    # ALL_ROWS checkpoint routes release their EXL3 tensors after conversion.
    # The live MXFP6 weight owns the device at this lifecycle boundary.
    device = state.merged_weight.values.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("rank64 bundle is currently verified only on SM120")
    rank = get_tensor_model_parallel_rank()
    entry = next(e for e in data["entries"] if (e["layer"], e["rank"]) == (int(match[1]), rank))
    setattr(layer, ATTRIBUTE, Rank64GateUp(load_entry(Path(path), entry), entry, device))
    return True


def requires_bf16(layer: Any, rows: int, norm: Any | None = None) -> bool:
    state = getattr(layer, ATTRIBUTE, None)
    if state is None:
        return False
    if norm is not None:
        state.verify_norm(norm)
    return rows == 32


def apply(layer: Any, x: Any, rows: int) -> torch.Tensor | None:
    if not requires_bf16(layer, rows):
        return None
    if not isinstance(x, torch.Tensor):
        raise RuntimeError("rank64 M32 received packed MXFP8; install the matching vLLM runtime patch")
    return getattr(layer, ATTRIBUTE).apply(x)
