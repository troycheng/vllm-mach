# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted for fixed TP2 GU packing and the deployed reciprocal-global convention.
"""Offline Hessian-selected NVFP4 packer for one original-BF16 merged GU.

Diagnostic-only: this deliberately has no live dispatch, graph, or automatic
vLLM quantizer call.  The selected E4M3 block scales are packed explicitly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


BLOCK = 16
N_GU = 17408
K_GU = 5120
GROUPS_GU = K_GU // BLOCK
G_NUMERATOR = 2688.0  # 448 E4M3 max times 6 E2M1 max.
HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
UPSTREAM_COMMIT = "51de53e48ccae8804f8fe1198b7cf89475c5c4f4"
UPSTREAM_SHA256 = {
    "nvfp4_fp8_sweep.py": "62e061579ab3db001660f40a29ccfa6958390b292a6ccc3af000e21a86d02506",
    "nvfp4_quant.py": "b57608fbd11f56750471d2d104fa3027910a28675492f9ddec263d15b189d02a",
    "_fp8_scale_candidates.py": "b0c1e1fc6cd3d748924a781a2c72ed16dfbf3dcbdab307fffb08e2944a88bc72",
    "fp4_kernel.py": "0b923c62402e2be68343ceaca75ebd1ac7aea28629d721a967d9473e5b991a4e",
}
E2M1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32)

try:  # Keep import and syntax checks usable on this CPU-only workstation.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only by the GPU pilot.
    triton = None
    tl = None


if triton is not None:
    # Extracted without algorithm changes from ModelOpt nvfp4_quant.py and
    # nvfp4_fp8_sweep.py at UPSTREAM_COMMIT; source hashes are above.
    @triton.jit
    def _fp4_round_magnitude(abs_scaled):
        return tl.where(
            abs_scaled <= 0.25,
            0.0,
            tl.where(
                abs_scaled < 0.75,
                0.5,
                tl.where(
                    abs_scaled <= 1.25,
                    1.0,
                    tl.where(
                        abs_scaled < 1.75,
                        1.5,
                        tl.where(
                            abs_scaled <= 2.5,
                            2.0,
                            tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, 6.0)),
                        ),
                    ),
                ),
            ),
        )

    @triton.jit
    def _hessian_sweep_kernel(
        x_ptr, hessian_ptr, candidate_scales_ptr, candidate_amaxes_ptr, best_amax_ptr,
        COUT, N_CIN_BLOCKS, BLOCK_SIZE: tl.constexpr, NUM_CANDIDATES: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        cin_block = pid % N_CIN_BLOCKS
        rows = (pid // N_CIN_BLOCKS) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
        row_mask = rows < COUT
        block_idx = rows * N_CIN_BLOCKS + cin_block
        elem = tl.arange(0, BLOCK_SIZE)
        w = tl.load(x_ptr + block_idx[:, None] * BLOCK_SIZE + elem[None, :],
                    mask=row_mask[:, None], other=0.0).to(tl.float32)
        w_abs = tl.abs(w)
        w_sign = tl.where(w >= 0, 1.0, -1.0)
        idx = tl.arange(0, BLOCK_SIZE)
        hessian = tl.load(hessian_ptr + cin_block * (BLOCK_SIZE * BLOCK_SIZE)
                          + idx[:, None] * BLOCK_SIZE + idx[None, :]).to(tl.float32)
        best_loss = tl.full([ROWS_PER_PROGRAM], float("inf"), dtype=tl.float32)
        best_idx = tl.zeros([ROWS_PER_PROGRAM], dtype=tl.int32)
        for candidate in tl.range(NUM_CANDIDATES):
            scale = tl.load(candidate_scales_ptr + candidate).to(tl.float32)
            scale_safe = tl.where(scale == 0.0, 1.0, scale)
            q_mag = _fp4_round_magnitude(w_abs / scale_safe)
            dw = w_sign * (w_abs - q_mag * scale_safe)
            hdw = tl.dot(dw, hessian, allow_tf32=False)
            loss = tl.sum(hdw * dw, axis=1)
            better = loss < best_loss
            best_loss = tl.where(better, loss, best_loss)
            best_idx = tl.where(better, candidate, best_idx)
        best_amax = tl.load(candidate_amaxes_ptr + best_idx, mask=row_mask, other=0.0).to(tl.float32)
        tl.store(best_amax_ptr + block_idx, best_amax, mask=row_mask)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_sources() -> dict[str, str]:
    actual = {name: _sha256(UPSTREAM / name) for name in UPSTREAM_SHA256}
    if actual != UPSTREAM_SHA256:
        raise RuntimeError(f"ModelOpt source identity mismatch: {actual}")
    return actual


def _fp8_candidates(device: torch.device) -> torch.Tensor:
    """Exact ModelOpt positive finite E4M3 candidate set divided by 448."""
    raw = torch.arange(128, dtype=torch.uint8, device=device)
    values = raw.view(torch.float8_e4m3fn).float()
    return values[torch.isfinite(values) & (values > 0)] / 448.0


def _best_amax(weight_nk: torch.Tensor, hessian: torch.Tensor, global_amax: torch.Tensor) -> torch.Tensor:
    """ModelOpt Hessian sweep result, laid out row-major (N, K/16)."""
    if triton is None:
        raise RuntimeError("Hessian NVFP4 sweep requires Triton in the GPU pilot environment")
    candidates = _fp8_candidates(weight_nk.device)
    # This is ModelOpt compute_fp4_scales for the candidate set.  The E4M3 cast
    # is intentional: it is the candidate scale used in dw^T H dw.
    candidate_amaxes = candidates * global_amax
    candidate_scales = (candidate_amaxes * (G_NUMERATOR / global_amax) / 6.0).to(
        torch.float8_e4m3fn
    ).float() / (G_NUMERATOR / global_amax)
    n, k = weight_nk.shape
    blocks = k // BLOCK
    best = torch.empty((n, blocks), dtype=torch.float32, device=weight_nk.device)
    grid = (triton.cdiv(n, 32) * blocks,)
    with torch.cuda.device(weight_nk.device):
        _hessian_sweep_kernel[grid](
            weight_nk.contiguous().view(-1), hessian.contiguous().view(-1),
            candidate_scales, candidate_amaxes, best.view(-1), n, blocks,
            BLOCK_SIZE=BLOCK, NUM_CANDIDATES=int(candidates.numel()),
            ROWS_PER_PROGRAM=32, num_warps=4,
        )
    return best


def _round_e2m1_codes(weight_nk: torch.Tensor, scale_n_g: torch.Tensor) -> torch.Tensor:
    """CPU/GPU torch mirror of upstream fp4_round_magnitude, then E2M1 codes."""
    zero_scale = scale_n_g == 0
    safe_scale = torch.where(zero_scale, torch.ones_like(scale_n_g), scale_n_g)
    scaled = weight_nk.float().abs().reshape(weight_nk.shape[0], -1, BLOCK) / safe_scale.unsqueeze(-1)
    magnitude = torch.where(scaled <= .25, 0, torch.where(scaled < .75, 1,
        torch.where(scaled <= 1.25, 2, torch.where(scaled < 1.75, 3,
        torch.where(scaled <= 2.5, 4, torch.where(scaled < 3.5, 5,
        torch.where(scaled <= 5., 6, 7))))))).to(torch.uint8)
    sign = (weight_nk.reshape(weight_nk.shape[0], -1, BLOCK) < 0).to(torch.uint8) << 3
    codes = magnitude | sign
    return torch.where(zero_scale.unsqueeze(-1), torch.zeros_like(codes), codes).reshape_as(weight_nk)


def _pack_low_even(codes_nk: torch.Tensor) -> torch.Tensor:
    return (codes_nk[:, 0::2] | (codes_nk[:, 1::2] << 4)).contiguous()


def _swizzle_sf_bytes(logical_bytes: torch.Tensor) -> torch.Tensor:
    """FlashInfer deployed 128x4 swizzle; rows/columns are padded before it."""
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    n, groups = logical_bytes.shape
    pn, pg = ((n + 127) // 128) * 128, ((groups + 3) // 4) * 4
    padded = torch.zeros((pn, pg), dtype=torch.uint8, device=logical_bytes.device)
    padded[:n, :groups] = logical_bytes
    physical = block_scale_interleave(padded)
    if physical.dtype != torch.uint8 or physical.numel() != pn * pg:
        raise RuntimeError("unexpected deployed FlashInfer 128x4 scale layout")
    return physical


def _inverse_swizzle_sf_bytes(physical_bytes: torch.Tensor, n: int, groups: int) -> torch.Tensor:
    """Invert the deployed swizzle with full padded-row labels, then truncate."""
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    pn, pg = ((n + 127) // 128) * 128, ((groups + 3) // 4) * 4
    flat = physical_bytes.contiguous().view(torch.uint8).flatten()
    if flat.numel() != pn * pg:
        raise ValueError(f"physical SF bytes={flat.numel()}, expected={pn * pg}")
    indices = torch.arange(pn * pg, dtype=torch.int64, device=flat.device).reshape(pn, pg)
    logical_index_for_physical = torch.zeros_like(indices)
    # uint8 labels are the documented FlashInfer input type. Three planes cover
    # this pilot's <2^24 padded scale positions without duplicate padding indices.
    for shift in (0, 8, 16):
        plane = ((indices >> shift) & 255).to(torch.uint8)
        logical_index_for_physical |= block_scale_interleave(plane).reshape(pn, pg).to(torch.int64) << shift
    logical = torch.empty_like(flat)
    logical[logical_index_for_physical.reshape(-1)] = flat
    return logical.reshape(pn, pg)[:n, :groups].contiguous()


def _make_weight(direct_module: Any, packed: torch.Tensor, scales: torch.Tensor,
                 global_scale: torch.Tensor, n: int, k: int) -> Any:
    cls = direct_module.DirectNVFP4Weight
    result = cls(packed=packed, scales=scales, global_scale=global_scale,
                 out_features=n, in_features=k)
    if hasattr(result, "validate"):
        result.validate()
    return result


def optimize_and_pack(weight_nk_bf16: torch.Tensor, hessian_fp32: torch.Tensor,
                      direct_module: Any) -> tuple[Any, dict[str, Any]]:
    """Select block scales with ModelOpt Hessian loss and explicitly pack NVFP4.

    ``weight_nk_bf16`` must be the original-checkpoint BF16 merged GU [17408,5120],
    not a K4/K5 Trellis reconstruction. ``hessian_fp32`` is [320,16,16] for the
    same input boundary. The function never calls ``scaled_fp4_quant``.
    """
    sources = _check_sources()
    if (weight_nk_bf16.dtype, tuple(weight_nk_bf16.shape), weight_nk_bf16.device.type) != (
        torch.bfloat16, (N_GU, K_GU), "cuda"
    ):
        raise ValueError("requires original-BF16 merged GU[N=17408,K=5120] on CUDA")
    if hessian_fp32.dtype != torch.float32 or tuple(hessian_fp32.shape) != (GROUPS_GU, BLOCK, BLOCK):
        raise ValueError("requires H_fp32[K/16=320,16,16] at the identical GU-input boundary")
    if not torch.isfinite(weight_nk_bf16).all() or not torch.isfinite(hessian_fp32).all():
        raise ValueError("non-finite weight or Hessian")
    hessian = hessian_fp32.to(device=weight_nk_bf16.device, dtype=torch.float32)
    global_amax = weight_nk_bf16.float().abs().amax().reshape(())
    if global_amax == 0:
        raise ValueError("zero GU weight is unsupported by the fixed global-scale contract")
    global_scale = (torch.tensor(G_NUMERATOR, dtype=torch.float32, device=weight_nk_bf16.device) / global_amax).reshape(1)
    best_amax = _best_amax(weight_nk_bf16, hessian, global_amax)
    # Do not call scaled_fp4_quant here: it would choose new absmax scales and
    # discard the Hessian-selected best_amax.  This is the fixed NVFP4 contract.
    scale_sf = (best_amax * global_scale / 6.0).to(torch.float8_e4m3fn)
    scale_n_g = scale_sf.float() / global_scale
    codes = _round_e2m1_codes(weight_nk_bf16, scale_n_g)
    packed = _pack_low_even(codes)
    physical_sf_bytes = _swizzle_sf_bytes(scale_sf.view(torch.uint8))
    # vLLM's ordinary scaled_fp4_quant exposes E4M3 SFB as [padded_N,K/16],
    # while B12X consumes its transpose. Retain that 2-D view and its raw bytes.
    physical_sf = physical_sf_bytes.reshape(N_GU, GROUPS_GU).view(torch.float8_e4m3fn)
    result = _make_weight(direct_module, packed, physical_sf, global_scale, N_GU, K_GU)
    receipt = {
        "schema": "hessian_nvfp4_original_bf16_merged_gu/v1", "hardware_verified": False,
        "adapter_sha256": _sha256(Path(__file__)), "upstream_commit": UPSTREAM_COMMIT,
        "upstream_sha256": sources,
        "direct_module_source": (
            {"path": str(direct_module.__file__), "sha256": _sha256(Path(direct_module.__file__))}
            if getattr(direct_module, "__file__", None) else None
        ),
        "shape_nk": [N_GU, K_GU], "block_size": BLOCK, "global_numerator": G_NUMERATOR,
        "global_amax": float(global_amax.item()), "global_scale": float(global_scale.item()),
        "best_amax_shape": [N_GU, GROUPS_GU], "scale_formula": "E4M3(best_amax*G_B/6)",
        "reconstruction": "E2M1*scale_sf/G_B", "scale_swizzle": "FlashInfer block_scale_interleave 128x4",
        "physical_scale_dtype": "uint8 bytes viewed as float8_e4m3fn", "physical_scale_shape": [N_GU, GROUPS_GU],
        "packed_nibble_order": "low=even K, high=odd K", "source_boundary": "original BF16 merged GU",
    }
    return result, receipt


def dequantize_nv4(weight: Any) -> torch.Tensor:
    """Diagnostic explicit E2M1 nibble decode plus inverse deployed swizzle.

    Supports padded physical SF rows, e.g. a logical [32,2560] activation-like
    weight with a physical [128,320] 128x4 scale buffer. It is not a hardware
    ABI validation; the GPU pilot independently compares FlashInfer decoding.
    """
    packed, scales, global_scale = weight.packed, weight.scales, weight.global_scale
    n, k = int(weight.out_features), int(weight.in_features)
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (n, k // 2) or k % BLOCK:
        raise ValueError("invalid packed E2M1 geometry")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise ValueError("invalid global scale")
    logical_sf = _inverse_swizzle_sf_bytes(scales.view(torch.uint8), n, k // BLOCK).view(torch.float8_e4m3fn)
    codes = torch.empty((n, k), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    values = E2M1.to(packed.device)[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return (values.reshape(n, -1, BLOCK) * logical_sf.float().unsqueeze(-1) / global_scale).reshape(n, k)
