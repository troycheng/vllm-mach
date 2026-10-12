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
    assert ssm.select_gdn(object()) is stock_gdn
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
