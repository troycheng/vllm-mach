"""Audit controls must restore timing paths and expose no callable RPC."""

import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("block_fp8", [False, True])
def test_unscored_audit_restores_original_methods(tmp_path, monkeypatch, block_fp8):
    spec = importlib.util.spec_from_file_location(
        "route_probe_test", Path(__file__).parents[1] / "tools/serving_route_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.delenv("MACH_BENCH_PROBE_DIR", raising=False)
    if block_fp8:
        monkeypatch.setenv("VLLM_MACH_PROFILE", "qwen35-4b-block-fp8-v1")
    else:
        monkeypatch.delenv("VLLM_MACH_PROFILE", raising=False)
    spec.loader.exec_module(module)

    class Hybrid:
        candidates = 128
        max_rows = 64
        backend = "b12x"

        def __init__(self):
            self.can_use_failure_counts = {}

        def coarse_logits(self, hidden):
            return "coarse"

        @staticmethod
        def refine_logits(hidden, weight, ids):
            return "refined"

    state = Hybrid()

    class Runner:
        def sample(self, hidden, batch, grammar):
            if block_fp8:
                return "refined"
            assert state.coarse_logits(hidden) == "coarse"
            return state.refine_logits(hidden, None, NS(shape=(4, 128)))

    runner = Runner()
    runner.model = NS(
        lm_head=NS(
            weight=NS(dtype="bfloat16", shape=(248320, 8)),
            _hybrid_nvfp4_lm_head_state=None if block_fp8 else state,
        )
    )
    runner.cudagraph_manager = NS(_capture_descs={})
    runner.model.modules = list
    worker = NS(rank=0, model_runner=runner)
    monkeypatch.setitem(
        sys.modules,
        "vllm_mach.mxfp6.hybrid_nvfp4_lm_head",
        None if block_fp8 else NS(HybridNvfp4LmHead=Hybrid),
    )
    monkeypatch.setenv("MACH_BENCH_PROBE_DIR", str(tmp_path))
    saved_coarse, saved_refine = Hybrid.coarse_logits, Hybrid.refine_logits
    module.install(worker)
    address = json.loads((tmp_path / "rank-0.json").read_text())["socket"]

    def call(op):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(address)
            client.sendall(json.dumps({"op": op}).encode() + b"\n")
            return json.loads(client.makefile("rb").readline())

    try:
        assert not call("snapshot")["audit_active"]
        assert "error" in call("arbitrary_callable")
        assert call("audit_on")["audit_active"]
        hidden = NS(shape=(4, 8))
        batch = NS(num_reqs=4, is_prefilling_np=NS(any=lambda: False))
        assert runner.sample(hidden, batch, None) == "refined"
        result = call("audit_off")
        assert not result["audit_active"]
        if block_fp8:
            assert result["counts"]["decode_complete_head_m4"] == 1
        else:
            assert result["counts"]["decode_hybrid_m4"] == 1
            assert result["counts"]["refine_m4_k128"] == 1
        assert Hybrid.coarse_logits is saved_coarse
        assert Hybrid.refine_logits is saved_refine
        assert runner.sample.__func__ is Runner.sample
        runner.sample(hidden, batch, None)
        assert call("snapshot")["counts"] == result["counts"]
    finally:
        worker._mach_bench_route_server.shutdown()
        worker._mach_bench_route_server.server_close()
        Path(address).unlink(missing_ok=True)
