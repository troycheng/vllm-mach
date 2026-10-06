"""CPU contracts for ordered state routing; no torch, vLLM, or GPU required."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

GDN = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/gdn"


def load_worker():
    spec = importlib.util.spec_from_file_location("mach_gdn_cpu_worker", GDN / "worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(dtype):
    ns = types.SimpleNamespace
    return ns(vllm_config=ns(
        parallel_config=ns(tensor_parallel_size=1), speculative_config=None,
        cache_config=ns(enable_prefix_caching=False, mamba_cache_mode="none",
                        use_replayssm=False, cache_dtype=dtype,
                        kv_cache_memory_bytes=19 * 2**30),
        model_config=ns(dtype="bf16", max_model_len=8192),
        scheduler_config=ns(max_num_seqs=128), kv_transfer_config=None))


class OrderedGDNContracts(unittest.TestCase):
    def test_import_does_not_load_gpu_dependencies(self):
        source = GDN.parents[2]
        program = (f"import sys; sys.path.insert(0, {str(source)!r}); "
                   "import vllm_mach.mxfp8.gdn; "
                   "assert not any(x in sys.modules for x in ('torch','vllm','triton'))")
        subprocess.run([sys.executable, "-c", program], check=True)

    def test_production_geometry_and_fp8_kv_contract(self):
        worker = load_worker()
        torch = types.ModuleType("torch")
        torch.bfloat16 = "bf16"
        with patch.dict(sys.modules, {"torch": torch}):
            for dtype in ("auto", "bfloat16", "fp8_e4m3"):
                worker._check_config(config(dtype))
            for field, value in (("max_num_seqs", 64),):
                candidate = config("fp8_e4m3")
                setattr(candidate.vllm_config.scheduler_config, field, value)
                with self.assertRaisesRegex(RuntimeError, "maxseq128"):
                    worker._check_config(candidate)
            for attr, value in (("enable_prefix_caching", True), ("mamba_cache_mode", "all"),
                                ("use_replayssm", True), ("kv_offloading_size", 1),
                                ("cache_dtype", "fp8_e5m2")):
                candidate = config("auto")
                setattr(candidate.vllm_config.cache_config, attr, value)
                with self.assertRaises(RuntimeError):
                    worker._check_config(candidate)
            candidate = config("auto")
            candidate.vllm_config.cache_config.kv_cache_memory_bytes = 4*2**30
            with self.assertRaises(RuntimeError):
                worker._check_config(candidate)

    def test_quality_mode_requires_explicit_fixed_rows_and_budget(self):
        worker = load_worker()
        torch = types.ModuleType("torch")
        torch.bfloat16 = "bf16"
        names = ("VLLM_MACH_MXFP8_MODE", "VLLM_MACH_MXFP8_QUALITY_ROWS")
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(worker._mode_contract(), ("production", None, 19*2**30))
            for rows in (32, 64):
                with patch.dict(os.environ, {names[0]: "quality", names[1]: str(rows)}):
                    candidate = config("fp8_e4m3")
                    budget = (4 if rows == 32 else 19)*2**30
                    candidate.vllm_config.cache_config.kv_cache_memory_bytes = budget
                    candidate.vllm_config.model_config.max_model_len = 1024
                    candidate.vllm_config.scheduler_config.max_num_seqs = rows
                    candidate.vllm_config.scheduler_config.max_num_batched_tokens = rows*256
                    worker._check_config(candidate)
                    self.assertEqual(worker._mode_contract(), ("quality", rows, budget))
                    with self.assertRaisesRegex(RuntimeError, f"maxseq{rows}/maxlen1024"):
                        worker._check_config(config("fp8_e4m3"))
                    candidate.vllm_config.scheduler_config.max_num_batched_tokens = 2048
                    with self.assertRaisesRegex(RuntimeError, f"maxbatch{rows*256}"):
                        worker._check_config(candidate)
            for env in ({names[0]: "quality"}, {names[0]: "quality", names[1]: "16"},
                        {names[1]: "32"}, {names[0]: "unknown"}):
                with patch.dict(os.environ, env):
                    with self.assertRaises(RuntimeError):
                        worker._check_config(config("auto"))
            worker._RUN_MODE, worker._QUALITY_ROW = "production", None
            with patch.dict(os.environ, {names[0]: "quality", names[1]: "32"}):
                candidate = config("auto")
                candidate.vllm_config.cache_config.kv_cache_memory_bytes = 4*2**30
                with self.assertRaisesRegex(RuntimeError, "mode changed"):
                    worker._check_config(candidate)
        manifest = json.loads((GDN.parent / "data/runtime_sources.json").read_text())
        self.assertEqual(worker.RUNNER_SHA,
            manifest["files"]["vllm/v1/worker/gpu/model_runner.py"]["installed_sha256"])
        self.assertEqual(worker.GDN_SHA,
            manifest["files"]["vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"]["installed_sha256"])

    def test_hook_composition_and_cache_phase_order(self):
        worker = load_worker()
        events = []
        class Worker:
            def init_device(self): events.append("device"); return "device"
            def load_model(self): events.append("load"); return "load"
            def initialize_from_config(self): events.append("cache"); return "cache"
            def compile_or_warm_up_model(self): events.append("compile"); return "compile"
        api = types.ModuleType("vllm.v1.worker.gpu_worker")
        api.Worker = Worker
        worker.install_runtime = lambda obj: events.append("install_runtime")
        worker._prepare_layers = lambda obj: events.append("prepare_layers")
        worker.initialize_cache = lambda obj: events.append("initialize_cache")
        worker._prepare_pools = lambda obj: events.append("prepare_pools")
        worker.inspect_worker = lambda obj: events.append("inspect")
        with patch.dict(sys.modules, {api.__name__: api}):
            self.assertTrue(worker.install_worker_hook())
            self.assertFalse(worker.install_worker_hook())
            obj = Worker()
            self.assertEqual(obj.init_device(), "device")
            self.assertEqual(obj.load_model(), "load")
            self.assertEqual(obj.initialize_from_config(), "cache")
            self.assertEqual(obj.compile_or_warm_up_model(), "compile")
        self.assertEqual(events, ["device", "install_runtime", "load", "prepare_layers",
                                  "cache", "initialize_cache", "prepare_pools", "compile", "inspect"])

    def test_strict_scheduler_slots_and_unique_worker_owner(self):
        worker = load_worker()
        self.assertEqual(worker._flatten_new(([0, 2, -1], [3, 2])), [2, 3, 2])
        self.assertEqual(worker._flatten_new(None), [])
        for invalid in ([[]], ((),), ([True],), (["1"],)):
            with self.assertRaises(TypeError): worker._flatten_new(invalid)
        owner = object()
        worker._claim_worker(owner)
        worker._claim_worker(owner)
        with self.assertRaisesRegex(RuntimeError, "another worker"):
            worker._claim_worker(object())
        with self.assertRaisesRegex(RuntimeError, "initialize"):
            worker.prepare_pools(owner)

    def test_kernel_arithmetic_and_launch_contracts(self):
        # AST fingerprints bind operations AND launch expressions; paths and
        # comments are excluded so normal packaging cannot invalidate the pin.
        fingerprints = {
            "lowm_allshape_triton": "4a02c9381d1c7cc6fc0732d0b92622e51afe758074ab6a4563faec78727911b3",
            "ordered_allm_triton": "8d288484698cf0aa4199d30e9ea6d400a89c57789dabb135885fbda2754c9e49",
            "ordered_m4_deferred_triton": "a55563f6c8ebdb0c553ecb96d02053561c26d66dd9186f2fdcfb129edb3b0020",
            "ordered_m16_deferred_triton": "73d585eb33e1fc3115f9317de4b166ffd3c35c1928ecf1ce8e4ac3869e7b5431",
        }
        for name, expected in fingerprints.items():
            tree = ast.parse((GDN / f"{name}.py").read_text())
            functions = [ast.dump(n, include_attributes=False) for n in tree.body
                         if isinstance(n, ast.FunctionDef) and
                         (n.name.startswith("_ordered_") or n.name in ("_lowm_ordered_decode", "decode"))]
            actual = hashlib.sha256("\n".join(functions).encode()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_formal_source_has_no_historical_path_or_control_selector(self):
        for path in GDN.glob("*.py"):
            source = path.read_text()
            for forbidden in ("/task/", "/legacy/", "_SELECTOR", "controlled=False", "algebraic_w4"):
                self.assertNotIn(forbidden, source, path.name)
            ast.parse(source)

    def test_stock_fallback_and_scheduler_reset_boundaries(self):
        tree = ast.parse((GDN / "worker.py").read_text())
        register = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_register_gdn")
        core = next(n for n in register.body if isinstance(n, ast.FunctionDef) and n.name == "core")
        statements = [ast.unparse(n) for n in core.body]
        materialize = next(i for i, text in enumerate(statements) if "component.materialize_slots" in text)
        self.assertIn("return _ORIG_CORE", statements[materialize + 1])
        lifecycle = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_install_runner_lifecycle")
        for name, predecessor in (("add", "old_add"), ("update", "old_update")):
            func = next(n for n in lifecycle.body if isinstance(n, ast.FunctionDef) and n.name == name)
            body = [ast.unparse(n) for n in func.body]
            original = next(i for i, text in enumerate(body) if predecessor + "(" in text)
            reset = next(i for i, text in enumerate(body) if "_reset_allocated(" in text)
            self.assertLess(original, reset)
        update = next(n for n in lifecycle.body if isinstance(n, ast.FunctionDef) and n.name == "update")
        self.assertIn("kv_cache_block_copies", ast.unparse(update.body[0]))


if __name__ == "__main__":
    unittest.main()
