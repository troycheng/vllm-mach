# SPDX-License-Identifier: Apache-2.0
"""Preserve the qualified Dense post-convolution Q/K reduction only."""
import functools
import hashlib
import importlib
import os
from pathlib import Path

_ENABLED = False
_INSTALLED = False
_STATS = {'calls': 0, 'capture_calls': 0, 'rows': {}}


def model_eligible(config):
    from ...runtime import attention

    if os.environ.get('VLLM_MACH_PROFILE') or not attention._NATIVE_REGISTERED:
        return False
    text = config.model_config.hf_text_config
    if (getattr(text, 'model_type', None) != 'qwen3_5_text'
            or not attention.preserve_old_softmax_segments(config)
            or config.cache_config.mamba_ssm_cache_dtype not in ('float32', 'float16')):
        return False
    parallel = config.parallel_config
    return (text.model_type == 'qwen3_5_text'
            and all(getattr(text, key, None) == value for key, value in
                    {'hidden_size': 5120, 'num_hidden_layers': 64,
                     'linear_num_key_heads': 16, 'linear_num_value_heads': 48,
                     'linear_key_head_dim': 128, 'linear_value_head_dim': 128}.items())
            and parallel.tensor_parallel_size == 2
            and parallel.pipeline_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and not parallel.enable_dbo and config.speculative_config is None)


def _sm120():
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and current_platform.is_device_capability(120)


def configure(config):
    global _ENABLED
    _ENABLED = model_eligible(config) and _sm120()
    if _ENABLED:
        install()


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch

    qwen = importlib.import_module(
        'vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn')
    original = qwen.fused_post_conv_prep

    @functools.wraps(original)
    def prep(conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
             apply_l2norm=True, output_g_exp=False):
        tensors = (conv_output, a, b, A_log, dt_bias)
        match = (_ENABLED and num_k_heads == 8 and head_k_dim == head_v_dim == 128
                 and conv_output.ndim == 2 and conv_output.shape[0] > 0
                 and conv_output.shape[1] == 5120
                 and conv_output.stride() == (5120, 1)
                 and a.shape == b.shape == (conv_output.shape[0], 24)
                 and a.stride() == b.stride() == (24, 1)
                 and A_log.shape == dt_bias.shape == (24,)
                 and A_log.stride() == dt_bias.stride() == (1,)
                 and conv_output.dtype == a.dtype == b.dtype == dt_bias.dtype == torch.bfloat16
                 and A_log.dtype == torch.float32
                 and all(t.is_cuda and t.device == conv_output.device for t in tensors)
                 and apply_l2norm is True and output_g_exp is False)
        implementation = original
        if match:
            from ..._gdn_prep_kernel import fused_post_conv_prep as implementation
        result = implementation(conv_output, a, b, A_log, dt_bias, num_k_heads,
                                head_k_dim, head_v_dim, apply_l2norm, output_g_exp)
        if match:
            _STATS['calls'] += 1
            rows = int(conv_output.shape[0])
            _STATS['rows'][rows] = _STATS['rows'].get(rows, 0) + 1
            if torch.cuda.is_current_stream_capturing():
                _STATS['capture_calls'] += 1
        return result

    qwen.fused_post_conv_prep = prep
    _INSTALLED = True


def receipt():
    helper = Path(__file__).parents[2] / '_gdn_prep_kernel.py'
    return dict(enabled=_ENABLED, installed=_INSTALLED, **_STATS,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                scope='Dense H8/HV24 prep only; FP32/FP16 state is not a prep operand',
                counters='host launches/captures, not CUDA graph replays')
