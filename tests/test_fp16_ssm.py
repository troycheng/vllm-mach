import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_mach.exl3 import ba_overlap as ba
from vllm_mach.exl3.mxfp6_hybrid import HybridRoute, HybridState

ROOT = Path(__file__).resolve().parents[1]
GDN = ROOT / 'profiles/flashinfer-0.6.18-gdn/overlay/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'


@pytest.fixture
def module(monkeypatch):
    monkeypatch.setenv('VLLM_MACH_BA_OVERLAP', '1')
    monkeypatch.setenv(ba.FP16_SSM_ENV, '1')
    monkeypatch.setenv(ba.EXTRA_ROWS_ENV, '1')
    monkeypatch.delenv('VLLM_MACH_BA_OVERLAP_VERIFY', raising=False)
    layer = NS(_mach_exl3_mxfp6=HybridState(HybridRoute.PREFILL_AND_M32, {}, object()),
               output_size_per_partition=64)
    return NS(_mach_fp16_ssm_decode_enabled=True,
              kv_cache=(NS(dtype=torch.bfloat16), NS(dtype=torch.float16)),
              tp_size=2, hidden_size=5120, num_k_heads=16, num_v_heads=48,
              head_k_dim=128, head_v_dim=128, enable_fused_gdn_decode=True,
              enable_packed_recurrent_decode=True, gqa_interleaved_layout=False,
              norm=NS(weight=NS(dtype=torch.bfloat16), activation='silu'),
              in_proj_qkvz=layer, in_proj_ba=layer,
              _fi_fused_decode_step=object(), _fi_fused_decode_supported=lambda *a, **kw: True,
              prefix='model.layers.0.linear_attn', conv_dim=8192, conv_kernel_size=4)


@pytest.mark.parametrize('rows,expected', [(4, False), (16, True), (24, True), (32, False), (128, False)])
def test_extra_rows(module, rows, expected):
    assert ba.extra_row_allowed(module, torch.empty(rows, 5120, dtype=torch.bfloat16)) is expected


@pytest.mark.parametrize('field,value', [('_mach_fp16_ssm_decode_enabled', False),
    ('tp_size', 1), ('hidden_size', 4096), ('num_k_heads', 8), ('num_v_heads', 32),
    ('head_k_dim', 64), ('head_v_dim', 64), ('enable_fused_gdn_decode', False),
    ('enable_packed_recurrent_decode', False), ('gqa_interleaved_layout', True),
    ('_fi_fused_decode_step', None)])
def test_extra_rows_reject_other_contracts(module, field, value):
    setattr(module, field, value)
    assert not ba.extra_row_allowed(module, torch.empty(16, 5120, dtype=torch.bfloat16))


@pytest.mark.parametrize('flag', [ba.FP16_SSM_ENV, ba.EXTRA_ROWS_ENV])
def test_explicit_flags(module, monkeypatch, flag):
    monkeypatch.delenv(flag)
    assert not ba.extra_row_allowed(module, torch.empty(16, 5120, dtype=torch.bfloat16))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_no_extra_rows_for_other_state_dtype(module, dtype):
    module.kv_cache[1].dtype = dtype
    assert not ba.extra_row_allowed(module, torch.empty(16, 5120, dtype=torch.bfloat16))


@pytest.mark.parametrize('route', [HybridRoute.ALL_ROWS, HybridRoute.PREFILL_ONLY])
def test_requires_original_exl3_decode_route(module, route):
    module.in_proj_qkvz._mach_exl3_mxfp6 = HybridState(route, {}, object())
    assert not ba.extra_row_allowed(module, torch.empty(16, 5120, dtype=torch.bfloat16))


def gdn_guard():
    # Execute the actual overlay method without importing CUDA/vLLM on CPU.
    tree = ast.parse(GDN.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == '_fused_gdn_decode_unsupported_reason')
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    platform = NS(is_cuda=lambda: True, is_device_capability=lambda sm: sm == 120,
                  has_device_capability=lambda sm: sm <= 120)
    fake_torch = NS(float16=torch.float16, float32=torch.float32, bfloat16=torch.bfloat16,
                    ops=NS(_C=NS(fused_gdn_decode_post_conv_mtp=object())))
    scope = dict(os=os, torch=fake_torch, current_platform=platform,
                 FP16_SSM_OPT_IN_ENV=ba.FP16_SSM_ENV,
                 FUSED_GDN_STATE_DTYPES=(torch.float32, torch.bfloat16))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(GDN), 'exec'), scope)
    return scope[fn.name], platform


@pytest.mark.parametrize('dtype,optin,accepted', [
    (torch.float16, True, True), (torch.float16, False, False),
    (torch.float32, False, True), (torch.bfloat16, False, True)])
def test_gdn_state_admission(module, monkeypatch, dtype, optin, accepted):
    guard, _ = gdn_guard()
    monkeypatch.setenv(ba.FP16_SSM_ENV, '1' if optin else '0')
    module.get_state_dtype = lambda: (torch.bfloat16, dtype)
    config = NS(model_config=NS(dtype=torch.bfloat16), speculative_config=None)
    assert (guard(module, config) is None) is accepted


@pytest.mark.parametrize('case', ['gpu', 'speculation', 'model_dtype', 'conv_dtype', 'activation'])
def test_fp16_guard_fail_closed(module, case):
    guard, platform = gdn_guard()
    module.get_state_dtype = lambda: (torch.bfloat16, torch.float16)
    config = NS(model_config=NS(dtype=torch.bfloat16), speculative_config=None)
    if case == 'gpu':
        platform.is_device_capability = lambda _: False
    elif case == 'speculation':
        config.speculative_config = object()
    elif case == 'model_dtype':
        config.model_config.dtype = torch.float16
    elif case == 'conv_dtype':
        module.get_state_dtype = lambda: (torch.float16, torch.float16)
    else:
        module.norm.activation = 'relu'
    assert guard(module, config) is not None


@pytest.mark.parametrize('rows,fi_supported,allowed', [(16, True, True), (24, False, True),
                                                     (32, True, False), (32, False, True)])
def test_before_qkv_fi_geometry_and_fp16_fallback(module, monkeypatch, rows, fi_supported, allowed):
    hidden = torch.empty(rows, 5120, dtype=torch.bfloat16)
    meta = NS(spec_sequence_masks=None, num_prefills=0, num_decodes=rows, num_actual_tokens=rows)
    context = NS(attn_metadata={module.prefix: meta})
    monkeypatch.setitem(sys.modules, 'vllm.forward_context', NS(get_forward_context=lambda: context))
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.layers.mamba.mamba_utils',
                        NS(is_conv_state_dim_first=lambda: True))
    module._fi_fused_decode_supported = lambda *a, **kw: fi_supported
    stream = NS(wait_stream=lambda _: None)
    monkeypatch.setattr(ba, '_AUX', stream)
    monkeypatch.setattr(ba, '_WARMED', True)
    monkeypatch.setattr(ba, '_EXTRA_WARMED_ROWS', {16, 24})
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: stream)
    assert (ba.before_qkv(module, hidden) is not None) is allowed
    meta.num_prefills = 1
    assert ba.before_qkv(module, hidden) is None
    meta.num_prefills = 0
    meta.num_actual_tokens = rows - 1
    assert ba.before_qkv(module, hidden) is None


def test_previous_a8_overlay_upgrade_hash():
    import json
    hashes = json.loads((ROOT / 'profiles/flashinfer-0.6.18-gdn/base-hashes.json').read_text())
    assert 'b0ea974fcdb008ce52df58cc1e80ea5c4c32a22b44a38a262b860c9c558a9bd5' in hashes[
        'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py']
