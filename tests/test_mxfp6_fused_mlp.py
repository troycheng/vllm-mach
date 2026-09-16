"""Compatibility when the installed 0.2.1 extension predates the new entry."""
from types import SimpleNamespace

import pytest
import torch

from vllm_mach.mxfp6 import fused_mlp


@pytest.mark.parametrize('mode', ['auto', '1', '0'])
def test_legacy_extension_selection(monkeypatch, mode):
    monkeypatch.setenv('VLLM_MACH_FUSED_SWIGLU_QUANT', mode)
    monkeypatch.setattr(fused_mlp, '_eligible', lambda module: True)
    monkeypatch.setattr(fused_mlp, '_import_mxfp6', lambda: SimpleNamespace(load_library=lambda: None))
    monkeypatch.setattr(fused_mlp, 'torch', SimpleNamespace(ops=SimpleNamespace(mxfp6=SimpleNamespace())))
    model = torch.nn.Module()
    if mode == '1':
        with pytest.raises(RuntimeError, match='gemm_from_swiglu'):
            fused_mlp.prepare(model)
    else:
        assert fused_mlp.prepare(model) == 0
    assert not hasattr(model, '_mach_swiglu_prepared')
