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

"""Fixed M32 dual-A4 stack preparer, preserving the selected quantizer bytes."""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - local source review.
    triton = None
    tl = None


M = 32
K = 5120
GROUP = 16
GROUPS = K // GROUP
STACKED_M = 2 * M
PADDED_M = 128
GROUPS_PER_CTA = 32


@dataclass(frozen=True)
class DualA4Buffers:
    """Capture-stable output buffers; padding is zeroed before graph capture."""

    packed: torch.Tensor
    scales: torch.Tensor


def make_buffers(device: torch.device) -> DualA4Buffers:
    """Allocate once; the hot path launches exactly one Triton kernel."""
    return DualA4Buffers(
        packed=torch.empty((STACKED_M, K // 2), dtype=torch.uint8, device=device),
        scales=torch.zeros((PADDED_M, GROUPS), dtype=torch.float8_e4m3fn, device=device),
    )


if triton is not None:
    @triton.jit
    def _rcp_approx_ftz(value):
        """The supplied upstream NVFP4 helper's PTX reciprocal operation."""
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _pack_e2m1_pair(low, high):
        """Upstream's RNE/saturating E2M1 pair pack: low nibble then high."""
        return tl.inline_asm_elementwise(
            "{ .reg .b8 byte; "
            "cvt.rn.satfinite.e2m1x2.f32 byte, $2, $1; "
            "cvt.u32.u8 $0, byte; }",
            "=r,f,f",
            [low, high],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _decode_e2m1(codes, sf, g):
        magnitude_code = (codes & 7).to(tl.int32)
        magnitude = tl.where(
            magnitude_code < 2,
            magnitude_code.to(tl.float32) * 0.5,
            ((magnitude_code & 1) + 2).to(tl.float32)
            * tl.exp2((magnitude_code // 2 - 2).to(tl.float32)),
        )
        magnitude_decoded = magnitude * sf / g
        signed = (magnitude_decoded.to(tl.uint32, bitcast=True) | ((codes & 8) << 28)).to(tl.float32, bitcast=True)
        # This is the visible decode used by dual_a4_rowstack_v2 before the
        # mandated BF16 residual boundary.
        return signed


    @triton.jit
    def _subtract_rn(a, b):
        return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=f,f,f", [a,b], dtype=tl.float32, is_pure=True, pack=1)


    @triton.jit
    def _scale_e4m3(amax, g):
        """Mirror nvfp4_utils.cuh's NVFP4 SF/output-scale operation order."""
        # `g * (amax * rcp.approx.ftz(6))`, followed by finite E4M3 saturation.
        raw = tl.minimum(g * (amax * _rcp_approx_ftz(6.0)), 448.0)
        sf_bits = raw.to(tl.float8e4nv)
        sf = sf_bits.to(tl.float32)
        output_scale = tl.where(
            sf != 0.0,
            _rcp_approx_ftz(sf * _rcp_approx_ftz(g)),
            0.0,
        )
        return sf_bits, sf, output_scale


    @triton.jit
    def _dual_a4_stack_kernel(
        x_ptr,
        ga1_ptr,
        logical_to_physical_ptr,
        out_packed_ptr,
        out_scales_u8_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        GROUP: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_PER_CTA: tl.constexpr,
    ):
        # One CTA handles 32 flattened M32/K16 groups: [32,8] adjacent pairs.
        # M*GROUPS/32 = 320 exact CTAs, avoiding 10,240 one-group CTAs.
        logical_group = tl.program_id(axis=0) * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
        row = logical_group // GROUPS
        group = logical_group - row * GROUPS
        pair = tl.arange(0, GROUP // 2)
        x_offset = row[:, None] * K + group[:, None] * GROUP + pair[None, :] * 2
        x_low = tl.load(x_ptr + x_offset).to(tl.float32)
        x_high = tl.load(x_ptr + x_offset + 1).to(tl.float32)
        ga1 = tl.load(ga1_ptr).to(tl.float32)

        first_amax = tl.maximum(tl.max(tl.abs(x_low), axis=1), tl.max(tl.abs(x_high), axis=1))
        first_fp8, first_sf, first_output_scale = _scale_e4m3(first_amax, ga1)
        first_packed = _pack_e2m1_pair(
            x_low * first_output_scale[:, None],
            x_high * first_output_scale[:, None],
        )
        first_low_code = first_packed & 15
        first_high_code = (first_packed >> 4) & 15
        first_decode_low = _decode_e2m1(first_low_code, first_sf[:, None], ga1)
        first_decode_high = _decode_e2m1(first_high_code, first_sf[:, None], ga1)

        # Preserve rowstack v2: subtraction -> BF16 -> FP32 -> exactly *4.
        residual4_low = _subtract_rn(x_low, first_decode_low).to(tl.bfloat16).to(tl.float32) * 4.0
        residual4_high = _subtract_rn(x_high, first_decode_high).to(tl.bfloat16).to(tl.float32) * 4.0
        second_amax = tl.maximum(
            tl.max(tl.abs(residual4_low), axis=1), tl.max(tl.abs(residual4_high), axis=1)
        )
        second_fp8, second_sf, second_output_scale = _scale_e4m3(second_amax, ga1)
        second_packed = _pack_e2m1_pair(
            residual4_low * second_output_scale[:, None],
            residual4_high * second_output_scale[:, None],
        )

        packed_offset = row[:, None] * (K // 2) + group[:, None] * (GROUP // 2) + pair[None, :]
        tl.store(out_packed_ptr + packed_offset, first_packed.to(tl.uint8))
        tl.store(out_packed_ptr + M * (K // 2) + packed_offset, second_packed.to(tl.uint8))

        first_physical = tl.load(logical_to_physical_ptr + logical_group).to(tl.int32)
        second_physical = tl.load(logical_to_physical_ptr + logical_group + M * GROUPS).to(tl.int32)
        # The caller passes E4M3 storage as uint8: this is a bit-store, not an
        # integer-to-FP8 conversion, preserving the actual operator's ABI.
        tl.store(out_scales_u8_ptr + first_physical, first_fp8.to(tl.uint8, bitcast=True))
        tl.store(out_scales_u8_ptr + second_physical, second_fp8.to(tl.uint8, bitcast=True))


def prepare_into(
    x: torch.Tensor,
    ga1: torch.Tensor,
    logical_to_physical: torch.Tensor,
    buffers: DualA4Buffers,
) -> dict[str, torch.Tensor | str]:
    """Build logical A1/A2 packed inputs in one fixed M32/K5120 launch.

    ``logical_to_physical`` is the precomputed int32 ``scale_index(64, 5120)``
    map from rowstack v2. It maps logical [64,320] scales into physical
    E4M3 [128,320] byte positions.
    """
    if triton is None:
        raise RuntimeError("dual A4 fused preparer requires Triton")
    if (x.dtype, tuple(x.shape), x.device.type, x.is_contiguous()) != (
        torch.bfloat16,
        (M, K),
        "cuda",
        True,
    ):
        raise ValueError("x must be contiguous CUDA BF16 [32,5120]")
    if (ga1.dtype, tuple(ga1.shape), ga1.device, ga1.numel()) != (
        torch.float32,
        (1,),
        x.device,
        1,
    ):
        raise ValueError("GA1 must be the bound CUDA FP32 scalar tensor [1]")
    if (logical_to_physical.dtype, tuple(logical_to_physical.shape), logical_to_physical.device) != (
        torch.int32,
        (STACKED_M, GROUPS),
        x.device,
    ):
        raise ValueError("logical_to_physical must be CUDA int32 scale_index(64,5120)")
    if (buffers.packed.dtype, tuple(buffers.packed.shape), buffers.packed.device) != (
        torch.uint8,
        (STACKED_M, K // 2),
        x.device,
    ):
        raise ValueError("packed output geometry/dtype mismatch")
    if (buffers.scales.dtype, tuple(buffers.scales.shape), buffers.scales.device) != (
        torch.float8_e4m3fn,
        (PADDED_M, GROUPS),
        x.device,
    ):
        raise ValueError("scale output must be physical E4M3 [128,320]")
    _dual_a4_stack_kernel[((M * GROUPS) // GROUPS_PER_CTA,)](
        x,
        ga1,
        logical_to_physical,
        buffers.packed,
        buffers.scales.view(torch.uint8),
        M=M, K=K, GROUP=GROUP, GROUPS=GROUPS,
        GROUPS_PER_CTA=GROUPS_PER_CTA,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return {
        "a_packed": buffers.packed,
        "a_scales": buffers.scales,
        "a_global_scale": ga1,
        "stack_contract": "rows 0:32=A1; rows 32:64=4*BF16(X-decode(A1))",
    }
