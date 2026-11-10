# SPDX-License-Identifier: Apache-2.0
"""Block-only current-FLA selection and shared old-order post-convolution prep."""
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import MethodType

from .profile import NAME
_DATA = Path(__file__).with_name('data')
_INSTALLED = False
_HITS = {}
_ACTUAL = {}
_UNKNOWN = {}
_PREP = 0
_STOCK_PREP = None


def fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, num_k_heads,
                         head_k_dim, head_v_dim, apply_l2norm=True, output_g_exp=False):
    global _PREP
    import torch
    eligible = (os.environ.get('VLLM_MACH_PROFILE') == NAME and num_k_heads == 16
                and A_log.numel() == 32 and head_k_dim == head_v_dim == 128
                and conv_output.is_cuda and conv_output.ndim == 2
                and conv_output.dtype == torch.bfloat16 and conv_output.shape[1] == 8192
                and apply_l2norm is True and output_g_exp is False)
    if eligible:
        from vllm_mach._gdn_prep_kernel import fused_post_conv_prep as implementation
        _PREP += 1
    else:
        implementation = _STOCK_PREP
    return implementation(conv_output, a, b, A_log, dt_bias, num_k_heads,
                          head_k_dim, head_v_dim, apply_l2norm, output_g_exp)


def choose(tuner, row, values):
    """Exact autotune key, including original argument-order dtype suffix."""
    from triton import Config
    bound = {k: v for k, v in values.items() if k in tuner.arg_names}
    key = [bound[k] for k in tuner.keys if k in bound]
    key.extend(str(v.dtype) for v in bound.values() if hasattr(v, 'dtype'))
    if key != row['key']:
        _UNKNOWN.setdefault(row['function'], [])
        if key not in _UNKNOWN[row['function']]:
            _UNKNOWN[row['function']].append(key)
        return None
    config = Config(**row['selected'])
    if config not in tuner.configs:
        raise RuntimeError('Qualified block-FP8 FLA Config disappeared')
    if tuple(key) in tuner.cache and tuner.cache[tuple(key)] != config:
        raise RuntimeError('Block-FP8 FLA selection installed after conflicting tuning')
    tuner.cache[tuple(key)] = config
    return key


def install():
    global _INSTALLED, _STOCK_PREP
    if os.environ.get('VLLM_MACH_PROFILE') != NAME:
        return False
    if _INSTALLED:
        return False
    from triton.runtime.autotuner import Autotuner
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep as stock
    if qwen.fused_post_conv_prep is not stock:
        raise RuntimeError('Block-FP8 GDN prep already replaced')
    gates = json.loads((_DATA / 'legacy_math_sources.json').read_text())['fla_sources']
    from vllm_mach import _gdn_prep_kernel
    if hashlib.sha256(Path(_gdn_prep_kernel.__file__).read_bytes()).hexdigest() != 'd574290a6b3dea264fe7a9f257be30647df9877038284e30c601b9add78dc30c':
        raise RuntimeError('Qualified shared GDN prep source changed')
    rows = json.loads((_DATA / 'fla_choices.json').read_text())['kernels']
    for row in rows:
        module = importlib.import_module(row['module'])
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if digest != gates[Path(module.__file__).name]:
            raise RuntimeError('Block-FP8 FLA source changed: ' + row['module'])
        tuner = getattr(module, row['function'])
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        original = tuner.run

        @functools.wraps(original)
        def run(self, *args, _original=original, _row=row, _digest=digest, **kwargs):
            values = {**dict(zip(self.arg_names, args)), **kwargs}
            key = choose(self, _row, values)
            result = _original(*args, **kwargs)
            if key is not None:
                name = _row['function']
                _HITS[name] = _HITS.get(name, 0) + 1
                _ACTUAL[name] = dict(key=key, selected=_row['selected'], source_sha256=_digest)
                if hasattr(result, 'hash'):
                    _ACTUAL[name].update(kernel_hash=result.hash, num_warps=result.metadata.num_warps,
                                         num_stages=result.metadata.num_stages)
            return result

        tuner.run = MethodType(run, tuner)
    _STOCK_PREP = stock
    qwen.fused_post_conv_prep = fused_post_conv_prep
    _INSTALLED = True
    return True


def inspect(worker=None):
    result = dict(installed=_INSTALLED, sixpin_calls=dict(_HITS), actual_choices=dict(_ACTUAL),
                  unknown_keys=dict(_UNKNOWN), shared_prep_calls=_PREP)
    if _INSTALLED:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
        from vllm_mach import _gdn_prep_kernel
        from .guard_norm import cache_receipt
        result['shared_prep_compiled'] = cache_receipt(_gdn_prep_kernel._fused_post_conv_kernel)
        result['shared_prep_source_sha256'] = hashlib.sha256(Path(_gdn_prep_kernel.__file__).read_bytes()).hexdigest()
        if qwen.fused_post_conv_prep is not fused_post_conv_prep:
            raise RuntimeError('Block-FP8 prep binding changed')
    if worker is not None:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        backends = [(name, m.gdn_prefill_backend, m.chunk_gated_delta_rule.gdn_prefill_backend)
                    for name, m in worker.get_model().named_modules() if isinstance(m, QwenGatedDeltaNetAttention)]
        if len(backends) != 24 or any(tuple(row[1:]) != ('triton', 'triton') for row in backends):
            raise RuntimeError('Block-FP8 requires its qualified Triton FLA prefill backend')
        result['actual_backends'] = backends
    return result
