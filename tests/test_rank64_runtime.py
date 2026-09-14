from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    # Test the standalone sidecar without importing vLLM's model registry.
    root = Path(__file__).resolve().parents[1] / "src/vllm_mach/exl3"
    package = ModuleType("_mach_rank64_test")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + ".mxfp6_hybrid", ModuleType("hybrid"))
    spec = importlib.util.spec_from_file_location(package.__name__ + ".rank64", root / "rank64.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_disabled_does_not_initialize_cuda(runtime, monkeypatch):
    monkeypatch.delenv(runtime.ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: pytest.fail("CUDA queried"))
    assert runtime.attach_layer(SimpleNamespace()) is False


def test_attach_after_checkpoint_releases_exl3_tensors(runtime, monkeypatch):
    monkeypatch.setenv(runtime.ENV, "/bundle")
    monkeypatch.setenv("VLLM_MACH_EXL3_MXFP6_FUSED_AR_NORM_MXFP8", "1")
    entry = {"layer": 0, "rank": 1}
    monkeypatch.setattr(runtime, "_bundle", lambda _: {"mask": [0], "entries": [entry]})
    device = torch.device("cuda:1")
    hybrid = SimpleNamespace(merged_weight=SimpleNamespace(values=SimpleNamespace(device=device)))
    monkeypatch.setattr(runtime.mxfp6_hybrid, "state_for_rows", lambda *args: hybrid, raising=False)
    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 1, get_tensor_model_parallel_world_size=lambda: 2))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (12, 0))
    monkeypatch.setattr(runtime, "load_entry", lambda root, e: {"loaded": True})
    monkeypatch.setattr(runtime, "Rank64GateUp", lambda tensors, e, d: (tensors, e, d))
    layer = SimpleNamespace(prefix="language_model.model.layers.0.mlp.gate_up_proj",
                            trellis=SimpleNamespace(exl3_tensors={}))
    assert runtime.attach_layer(layer)
    assert getattr(layer, runtime.ATTRIBUTE) == ({"loaded": True}, entry, device)


def test_m32_only_and_packed_input_rejected(runtime):
    state = SimpleNamespace(apply=lambda x: x)
    layer = SimpleNamespace(**{runtime.ATTRIBUTE: state})
    for rows in (1, 2, 4, 8, 16, 24, 33, 3000, 4096):
        assert not runtime.requires_bf16(layer, rows)
        assert runtime.apply(layer, object(), rows) is None
    assert runtime.requires_bf16(layer, 32)
    with pytest.raises(RuntimeError, match="packed MXFP8"):
        runtime.apply(layer, object(), 32)


def test_norm_check_before_capture(runtime, monkeypatch):
    state = runtime.Rank64GateUp.__new__(runtime.Rank64GateUp)
    norm = SimpleNamespace(weight=torch.zeros(5120, dtype=torch.bfloat16), variance_epsilon=1e-6)
    state.norm_verified = False
    state.entry = {"norm_bf16_sha256": runtime.tensor_sha256(norm.weight)}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    state.verify_norm(norm)
    assert state.norm_verified
    state.norm_verified = False
    norm.weight[0] = 1
    with pytest.raises(ValueError, match="another RMSNorm"):
        state.verify_norm(norm)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager forward"):
        state.verify_norm(norm)


def test_fork_join_and_gemm_arguments(runtime, monkeypatch):
    events = []
    current = SimpleNamespace(wait_stream=lambda stream: events.append("join"))
    aux = SimpleNamespace(wait_stream=lambda stream: events.append("fork"))

    @contextlib.contextmanager
    def stream_context(stream):
        assert stream is aux
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch, "mm", lambda *args, **kwargs: events.append("mm"))
    state = runtime.Rank64GateUp.__new__(runtime.Rank64GateUp)
    state.packed = torch.empty((2, 2), dtype=torch.uint8)
    state.scales = torch.empty((2, 2), dtype=torch.float8_e4m3fn)
    state.a = state.b = state.z = object()
    state.a_global_scale = state.index = state.buffers = object()
    state.alpha = torch.ones(1)
    state.aux_stream = aux
    state.graph_rows = set()
    packed_a, scales_a, terms, output = object(), object(), object(), object()

    def prepare(*args):
        events.append("prepare")
        return {"a_packed": packed_a, "a_scales": scales_a}

    def gemm(a, b, sa, sb, alpha, **kwargs):
        events.append("gemm")
        assert a is packed_a and sa is scales_a and alpha is state.alpha
        assert b.data_ptr() == state.packed.data_ptr() and sb.data_ptr() == state.scales.data_ptr()
        assert kwargs == dict(out_dtype=torch.bfloat16, block_size=16, use_8x4_sf_layout=False,
                              backend="b12x", use_nvfp4=True, enable_pdl=False)
        return terms

    def merge(t, z, b):
        assert t is terms and z is state.z and b is state.b
        events.append("merge")
        return output

    state._prepare, state._gemm = prepare, gemm
    monkeypatch.setitem(sys.modules, "_mach_rank64_test.rank64_merge", SimpleNamespace(up_merge=merge))
    assert state.apply(torch.zeros((32, 5120), dtype=torch.bfloat16)) is output
    assert events == ["fork", "enter", "mm", "exit", "prepare", "gemm", "join", "merge"]
    assert state.graph_rows == {32}


def test_rank64_gate_rejects_wrong_rows_before_cuda(runtime, monkeypatch):
    state = runtime.Rank64GateUp.__new__(runtime.Rank64GateUp)
    # Stub the lazy merge import; the shape guard must execute before any op.
    monkeypatch.setitem(sys.modules, "_mach_rank64_test.rank64_merge", SimpleNamespace(up_merge=None))
    with pytest.raises(ValueError, match="BF16"):
        state.apply(torch.zeros((24, 5120), dtype=torch.bfloat16))
