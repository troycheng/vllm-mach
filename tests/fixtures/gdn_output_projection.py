# SPDX-License-Identifier: Apache-2.0
"""Pinned shared GDN method before the 4B-only frontend hook."""
from __future__ import annotations

class LegacyQwenGatedDeltaNetAttention:
    def _output_projection(self, core_attn_out, z):
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        if current_platform.is_cuda() and core_attn_out.ndim == z.ndim == 3:
            # Keep the head axis through normalization: flattening it first
            # can make Inductor materialize an identity copy before out_proj.
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(-1, self.value_dim // self.tp_size)
        else:
            z_shape_og = z.shape
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output, _ = self.out_proj(core_attn_out)
        return output


class QwenGatedDeltaNetAttention:
    def _output_projection(self, core_attn_out, z):
        """The vLLM 0.33 output projection before the 4B hook."""
        core_attn_out = self.norm(core_attn_out, z)
        output, _ = self.out_proj(core_attn_out.flatten(-2))
        return output
