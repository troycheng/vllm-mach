"""Attention fusion stays restricted to native TP2 partial output projections."""
from types import SimpleNamespace

import pytest
import torch

from vllm_mach.mxfp6 import fused_attention
from vllm_mach.mxfp6.dense import Mxfp6Sm120LinearKernel


def module():
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    attn = object.__new__(Qwen3NextAttention)
    torch.nn.Module.__init__(attn)
    out = object.__new__(RowParallelLinear)
    torch.nn.Module.__init__(out)
    out.tp_size = 2
    out.input_is_parallel = True
    out.reduce_results = False
    out.bias = None
    out.weight = torch.empty(5120, 2304, dtype=torch.uint8, device='meta')
    out.scheme = SimpleNamespace(ocp_mx_linear=object.__new__(Mxfp6Sm120LinearKernel))
    attn.o_proj = out
    attn.attn_output_gate = True
    attn.q_size = 3072
    return attn


def test_native_eligibility():
    assert fused_attention._eligible(module())


@pytest.mark.parametrize('field,value', [('tp_size', 1), ('reduce_results', True),
    ('input_is_parallel', False), ('bias', torch.empty(1)), ('scheme', None)])
def test_projection_fallback(field, value):
    attn = module()
    setattr(attn.o_proj, field, value)
    assert not fused_attention._eligible(attn)


def test_gate_fallback():
    attn = module()
    attn.attn_output_gate = False
    assert not fused_attention._eligible(attn)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv('VLLM_MACH_FUSED_ATTN_QUANT', raising=False)
    monkeypatch.setattr(fused_attention, '_import_mxfp6', lambda: pytest.fail('loaded disabled producer'))
    assert fused_attention.prepare(module()) == 0


def test_invalid_mode(monkeypatch):
    monkeypatch.setenv('VLLM_MACH_FUSED_ATTN_QUANT', 'auto')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        fused_attention.prepare(module())


def test_prepare_keeps_unsupported_rows_on_original(monkeypatch):
    attn = module()
    calls = []
    attn.forward = lambda positions, hidden_states: calls.append(hidden_states.shape) or hidden_states
    monkeypatch.setenv('VLLM_MACH_FUSED_ATTN_QUANT', '1')
    monkeypatch.setattr(fused_attention, '_import_mxfp6',
                        lambda: SimpleNamespace(load_library=lambda: None))
    monkeypatch.setattr(fused_attention, '_REGISTERED', True)
    monkeypatch.setattr(torch.ops.mxfp6, 'gemm_w6a8_pdl', lambda *args: None, raising=False)
    assert fused_attention.prepare(attn) == 1
    assert fused_attention.prepare(attn) == 0
    # CPU, dtype and physical row guards must preserve the original forward.
    for rows, dtype in [(33, torch.bfloat16), (4, torch.float32), (4, torch.bfloat16)]:
        x = torch.empty(rows, 5120, dtype=dtype)
        assert attn(torch.arange(rows), x) is x
    assert len(calls) == 3
