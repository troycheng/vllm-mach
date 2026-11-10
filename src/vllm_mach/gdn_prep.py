# SPDX-License-Identifier: Apache-2.0
"""Explicit profile/geometry selection for the shared post-convolution helper."""
import os
import hashlib
from pathlib import Path

_PROFILE = 'qwen35-2b-mxfp8-champion-v1'


def identity():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name('_gdn_prep_kernel.py')]}


def four_b_enabled():
    return (os.environ.get('VLLM_MACH_PROFILE') == 'qwen35-4b-mxfp8-champion-v1'
            and os.environ.get('VLLM_MACH_MXFP8_SSM_DTYPE', 'float32') == 'bfloat16'
            and os.environ.get('VLLM_MACH_MXFP8_ORDERED_BF16_SSM', '1') == '1')


def profile_enabled():
    return os.environ.get('VLLM_MACH_PROFILE') == _PROFILE or four_b_enabled()


def verify_binding():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep as stock
    expected = fused_post_conv_prep if profile_enabled() else stock
    if qwen_gdn_linear_attn.fused_post_conv_prep is not expected:
        raise RuntimeError('GDN was imported before its explicit profile selection')


def selected(profile, *, heads, value_heads, head_dim, value_dim, bf16,
             apply_l2norm, output_g_exp):
    return (((profile == _PROFILE and heads == value_heads == 16)
             or (profile == 'qwen35-4b-mxfp8-champion-v1' and four_b_enabled()
                 and heads == 16 and value_heads == 32))
            and head_dim == value_dim == 128 and bf16
            and apply_l2norm is True and output_g_exp is False)


def fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, num_k_heads,
                         head_k_dim, head_v_dim, apply_l2norm=True,
                         output_g_exp=False):
    import torch
    from vllm.third_party.flash_linear_attention.ops import fused_gdn_prefill_post_conv as stock
    enabled = selected(os.environ.get('VLLM_MACH_PROFILE'), heads=num_k_heads,
                       value_heads=A_log.numel(), head_dim=head_k_dim,
                       value_dim=head_v_dim, bf16=conv_output.dtype == torch.bfloat16,
                       apply_l2norm=apply_l2norm, output_g_exp=output_g_exp)
    if enabled:
        from ._gdn_prep_kernel import fused_post_conv_prep as implementation
    else:
        implementation = stock.fused_post_conv_prep
    return implementation(conv_output, a, b, A_log, dt_bias, num_k_heads,
                          head_k_dim, head_v_dim, apply_l2norm, output_g_exp)
