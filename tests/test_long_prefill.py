from types import SimpleNamespace
import pytest
import torch
from vllm_mach.exl3 import long_prefill as lp


def test_exact_shapes_and_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.delenv('VLLM_MACH_LOSSLESS_PREFILL_LONG', raising=False)
    assert not lp.eligible((3000, 5120), torch.bfloat16, 2)
    monkeypatch.setenv('VLLM_MACH_LOSSLESS_PREFILL_LONG', '1')
    assert len(lp.ROWS) == 32
    for m in lp.ROWS:
        assert lp.eligible((m, 5120), torch.bfloat16, 2)
    for m in (32, 1023, 3004, 3276, 4096):
        assert not lp.eligible((m, 5120), torch.bfloat16, 2)
    assert not lp.eligible((3000, 5120), torch.float16, 2)
    assert not lp.eligible((3000, 5120), torch.bfloat16, 1)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    assert not lp.eligible((3000, 5120), torch.bfloat16, 2)


def test_workspace_capacity():
    for m in lp.ROWS:
        meta = dict(tp_size=2, tp_rank=0, hidden_dim=5120, max_token_num=m,
                    buffer_size=lp.workspace_bytes(m))
        ws = SimpleNamespace(backend='trtllm', metadata=meta)
        lp.validate_workspace(ws, 0, m)
        meta['buffer_size'] -= 1
        with pytest.raises(RuntimeError, match='capacity'):
            lp.validate_workspace(ws, 0, m)
