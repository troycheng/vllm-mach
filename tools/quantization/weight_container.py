# SPDX-License-Identifier: Apache-2.0
"""Original NVFP4 storage container; no Trellis conversion is included."""
import torch

class DirectNVFP4Weight:
    """One CUTLASS-layout NVFP4 weight produced directly from Trellis."""

    format = "nvfp4_group16_e4m3"

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        global_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> None:
        self.packed = packed
        self.scales = scales
        self.global_scale = global_scale
        self.out_features = out_features
        self.in_features = in_features
        self.format = type(self).format

    def validate(self) -> None:
        if self.packed.device.type != "cuda":
            raise ValueError("DirectNVFP4Weight must remain on CUDA")
        if self.scales.device != self.packed.device:
            raise ValueError("NVFP4 values/scales must share a device")
        if self.global_scale.device != self.packed.device:
            raise ValueError("NVFP4 global_scale must be a CUDA tensor")
        if self.global_scale.dtype != torch.float32 or self.global_scale.numel() != 1:
            raise ValueError("NVFP4 global_scale must be one FP32 CUDA value")
        if tuple(self.packed.shape) != (self.out_features, self.in_features // 2):
            raise ValueError(
                "NVFP4 packed weight shape mismatch: "
                f"got={tuple(self.packed.shape)}, "
                f"expected={(self.out_features, self.in_features // 2)}"
            )
        if self.in_features % 16:
            raise ValueError("NVFP4 K must be divisible by group size 16")
