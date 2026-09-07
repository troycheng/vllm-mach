from types import SimpleNamespace

import pytest
import torch

from vllm_mach.exl3.lossless_prefill import eligible, validate_workspace, MIN_WORKSPACE_BYTES, SUM_MIN_WORKSPACE_BYTES


def test_mode_selection(monkeypatch):
    from vllm_mach.exl3.lossless_prefill import selected_mode
    monkeypatch.delenv('VLLM_MACH_LOSSLESS_PREFILL_SUM', raising=False)
    monkeypatch.delenv('VLLM_MACH_LOSSLESS_PREFILL_DIRECT', raising=False)
    assert selected_mode() == 'input'
    monkeypatch.setenv('VLLM_MACH_LOSSLESS_PREFILL_DIRECT', '1')
    with pytest.raises(RuntimeError, match='Direct SUM requires'):
        selected_mode()
    monkeypatch.setenv('VLLM_MACH_LOSSLESS_PREFILL_SUM', '1')
    assert selected_mode() == 'direct'
    monkeypatch.setenv('VLLM_MACH_LOSSLESS_PREFILL_DIRECT', '0')
    assert selected_mode() == 'sum'


def test_opt_in_and_shape(monkeypatch):
    monkeypatch.delenv('VLLM_MACH_LOSSLESS_PREFILL', raising=False)
    assert not eligible((4096, 5120), torch.bfloat16, 2)
    monkeypatch.setenv('VLLM_MACH_LOSSLESS_PREFILL', '1')
    assert eligible((4096, 5120), torch.bfloat16, 2)
    for shape, dtype, tp in [((32, 5120), torch.bfloat16, 2),
                              ((4096, 4096), torch.bfloat16, 2),
                              ((4096, 5120), torch.float16, 2),
                              ((4096, 5120), torch.bfloat16, 4)]:
        assert not eligible(shape, dtype, tp)


def test_workspace_contract():
    meta = dict(tp_size=2, tp_rank=0, hidden_dim=5120, max_token_num=4096,
                buffer_size=MIN_WORKSPACE_BYTES)
    validate_workspace(SimpleNamespace(backend='trtllm', metadata=meta), 0)
    for key, value in [('tp_size', 4), ('tp_rank', 1), ('hidden_dim', 4096),
                       ('max_token_num', 4095), ('buffer_size', MIN_WORKSPACE_BYTES - 1)]:
        with pytest.raises(RuntimeError, match='mismatch'):
            validate_workspace(SimpleNamespace(backend='trtllm', metadata={**meta, key: value}), 0)
    with pytest.raises(RuntimeError, match='trtllm'):
        validate_workspace(SimpleNamespace(backend='mnnvl', metadata=meta), 0)


def test_sum_header_capacity():
    meta = dict(tp_size=2, tp_rank=0, hidden_dim=5120, max_token_num=4096,
                buffer_size=MIN_WORKSPACE_BYTES)
    workspace = SimpleNamespace(backend='trtllm', metadata=meta)
    validate_workspace(workspace, 0, sum_codec=False)
    with pytest.raises(RuntimeError, match='mismatch'):
        validate_workspace(workspace, 0, sum_codec=True)
    meta['buffer_size'] = SUM_MIN_WORKSPACE_BYTES
    validate_workspace(workspace, 0, sum_codec=True)
    meta['buffer_size'] -= 1
    with pytest.raises(RuntimeError, match='mismatch'):
        validate_workspace(workspace, 0, sum_codec=True)
