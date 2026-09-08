from types import SimpleNamespace

import torch

from vllm_mach.exl3 import ba_overlap as ba
from vllm_mach.exl3.mxfp6_hybrid import HybridRoute, HybridState


def test_disabled_does_not_touch_cuda(monkeypatch):
    monkeypatch.delenv('VLLM_MACH_BA_OVERLAP', raising=False)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kw: (_ for _ in ()).throw(AssertionError()))
    ba.prepare_stream()
    assert ba.before_qkv(object(), object()) is None


def test_mach_m32_route_contract():
    layer = SimpleNamespace()
    assert not ba.merged_m32_route(layer)
    for route in (HybridRoute.PREFILL_ONLY, HybridRoute.ALL_ROWS):
        layer._mach_exl3_mxfp6 = HybridState(route, {}, object())
        assert not ba.merged_m32_route(layer)
    layer._mach_exl3_mxfp6 = HybridState(HybridRoute.PREFILL_AND_M32, {}, None)
    assert not ba.merged_m32_route(layer)
    layer._mach_exl3_mxfp6 = HybridState(HybridRoute.PREFILL_AND_M32, {}, object())
    assert ba.merged_m32_route(layer)


def test_serial_fallbacks():
    hidden, packed = object(), object()
    module = SimpleNamespace(in_proj_ba=lambda x: (x, None), split_ba=lambda x: (x, x))
    assert ba.select_ba(module, hidden) == (hidden, None)
    assert ba.split_ba(module, packed) == (packed, packed)
    ba.before_recurrent(module)


def test_pending_buffers_and_join(monkeypatch):
    events = []
    main = SimpleNamespace(cuda_stream=7, wait_stream=lambda stream: events.append(stream))
    aux = object()
    packed, b, a = object(), object(), object()
    module = SimpleNamespace(_mach_ba_pending=dict(main=main, aux=aux, ba=packed, b=b, a=a))
    assert ba.select_ba(module, None) == (packed, None)
    assert ba.split_ba(module, packed) == (b, a)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: main)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    ba.before_recurrent(module)
    assert events == [aux]
    assert module._mach_ba_pending is None
