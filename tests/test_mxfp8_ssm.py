"""SSM selection must reach both vLLM configuration and real cache validation."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_mach.mxfp8 import ssm, stock_gdn


@pytest.mark.parametrize('model', ['2b', '4b'])
def test_launcher_dtype_resets_inherited_selection(monkeypatch, tmp_path, model):
    if model == '2b':
        from vllm_mach.mxfp8.two_b import serve, profile
    else:
        from vllm_mach.mxfp8 import serve, profile
    # Restore all settings populated by the launcher when the test ends.
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    serve.configure_environment(tmp_path, ssm_dtype='bfloat16')
    assert profile.run_contract()['ssm_dtype'] == 'bfloat16'
    argv = serve.build_argv('/model', ssm_dtype='bfloat16')
    assert argv[argv.index('--mamba-ssm-cache-dtype') + 1] == 'bfloat16'
    ordered = object()
    assert ssm.select_gdn(ordered) is ordered
    assert profile.run_contract()['ordered_gdn']
    serve.configure_environment(tmp_path, ssm_dtype='bfloat16', ordered_bf16_ssm=False)
    assert ssm.select_gdn(ordered) is stock_gdn
    serve.configure_environment(tmp_path)
    assert profile.run_contract()['ssm_dtype'] == 'float32'
    ordered = object()
    assert ssm.select_gdn(ordered) is ordered
    with pytest.raises(ValueError):
        serve.configure_environment(tmp_path, ssm_dtype='float16')


def test_pool_receipt_checks_actual_dtype(monkeypatch):
    monkeypatch.setattr(stock_gdn, '_check_unpatched', lambda: None)
    def tensor(dtype):
        return NS(dtype=dtype, is_cuda=True, ndim=4, shape=(10, 32, 128, 128),
                  stride=lambda: (524288, 16384, 128, 1))
    layer = NS(prefix='layer', kv_cache=(tensor(torch.bfloat16), tensor(torch.bfloat16)))
    monkeypatch.setattr(stock_gdn, '_layers', lambda worker: [layer])
    worker = NS(_mach_bf16_stock_gdn=dict(layers=['layer'], ready=True, cache_initialized=True))
    receipt = stock_gdn.inspect_worker(worker)
    assert receipt['state_dtype'] == 'bfloat16'
    assert receipt['ordered_hooks_installed'] is False
    layer.kv_cache[1].dtype = torch.float32
    with pytest.raises(RuntimeError, match='dtype/layout'):
        stock_gdn.inspect_worker(worker)


def test_stock_rejects_ordered_hooks(monkeypatch):
    import sys
    layer_path = 'vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn'
    runner_path = 'vllm.v1.worker.gpu.model_runner'
    gdn = NS(_mach_mxfp8_ordered_installed=True)
    monkeypatch.setitem(sys.modules, layer_path, NS(QwenGatedDeltaNetAttention=gdn))
    monkeypatch.setitem(sys.modules, runner_path, NS(GPUModelRunner=NS()))
    with pytest.raises(RuntimeError, match='cannot coexist'):
        stock_gdn._check_unpatched()


@pytest.mark.parametrize('ready,initialized', [(True, True), (False, True)])
def test_stock_rejects_live_cache_reallocation(ready, initialized):
    worker = NS(_mach_bf16_stock_gdn=dict(ready=ready, cache_initialized=initialized))
    with pytest.raises(RuntimeError, match='reallocation'):
        stock_gdn.check_cache_initialization(worker)


@pytest.mark.parametrize('model', ['2b', '4b'])
def test_ordered_bf16_default_and_explicit_stock_are_recorded(monkeypatch, tmp_path, model):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    if model == '2b':
        from vllm_mach.mxfp8.two_b import serve, profile
    else:
        from vllm_mach.mxfp8 import serve, profile
    ordered = object()
    monkeypatch.setenv(ssm.ENV, 'bfloat16')
    monkeypatch.delenv(ssm.ORDERED_BF16_ENV, raising=False)
    assert ssm.select_gdn(ordered) is ordered
    serve.configure_environment(tmp_path, ssm_dtype='bfloat16', ordered_bf16_ssm=True)
    assert ssm.select_gdn(ordered) is ordered
    contract = profile.run_contract()
    assert contract['ordered_gdn'] and contract['ssm_dtype'] == 'bfloat16'
    serve.configure_environment(tmp_path, ssm_dtype='bfloat16', ordered_bf16_ssm=False)
    assert ssm.select_gdn(ordered) is stock_gdn
    assert profile.run_contract() != contract
    monkeypatch.setenv(ssm.ORDERED_BF16_ENV, 'invalid')
    with pytest.raises(ValueError):
        ssm.select_gdn(ordered)


@pytest.mark.parametrize('model', ['2b', '4b'])
def test_ordered_layout_checks_selected_state_dtype(monkeypatch, model):
    if model == '2b':
        from vllm_mach.mxfp8.two_b.gdn import worker
        heads = 16
    else:
        from vllm_mach.mxfp8.gdn import worker
        heads = 32
    state = NS(dtype=torch.bfloat16, shape=(10, heads, 128, 128),
               stride=lambda *args: ((heads*16384, 16384, 128, 1)[args[0]]
                                     if args else (heads*16384, 16384, 128, 1)))
    layer = NS(tp_size=1, gqa_interleaved_layout=False,
               enable_packed_recurrent_decode=True, _is_sm120=True,
               num_k_heads=16, num_v_heads=heads, head_k_dim=128, head_v_dim=128,
               kv_cache=(NS(dtype=torch.bfloat16), state),
               A_log=NS(dtype=torch.float32), dt_bias=NS(dtype=torch.bfloat16))
    monkeypatch.setenv(ssm.ENV, 'bfloat16')
    assert worker._layout(layer, state)
    monkeypatch.setenv(ssm.ENV, 'float32')
    assert not worker._layout(layer, state)
