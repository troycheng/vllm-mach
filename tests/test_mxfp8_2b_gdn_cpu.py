"""CPU lifecycle contracts for the frozen 2B ordered GDN implementation."""
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

GDN = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/two_b/gdn"
PACKAGE = "mach_2b_gdn_cpu"
NS = types.SimpleNamespace


def load_worker():
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.worker", GDN / "worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(dtype="auto"):
    return NS(vllm_config=NS(
        parallel_config=NS(tensor_parallel_size=1), speculative_config=None,
        cache_config=NS(enable_prefix_caching=False, mamba_cache_mode="none",
                        use_replayssm=False, cache_dtype=dtype,
                        kv_cache_memory_bytes=19 * 2**30),
        model_config=NS(dtype="bf16", max_model_len=16384,
                        hf_text_config=NS(hidden_size=2048, num_hidden_layers=24,
                                          linear_num_key_heads=16, linear_num_value_heads=16,
                                          linear_key_head_dim=128, linear_value_head_dim=128)),
        scheduler_config=NS(max_num_seqs=160), kv_transfer_config=None))


def owner_api():
    api = types.ModuleType("vllm.v1.worker.worker_base")
    class WorkerWrapperBase:
        def __init__(self, worker): self.worker = worker
    api.WorkerWrapperBase = WorkerWrapperBase
    return api


def reset_api(events):
    torch = types.ModuleType("torch")
    torch.int32, torch.bfloat16, torch.float32 = "i32", "bf16", "f32"
    torch.device = lambda device: device
    class HostIDs:
        def __init__(self, ids): self.ids = tuple(ids)
        def to(self, device, *, non_blocking):
            events.append(("h2d", self, device, non_blocking))
            return NS(ids=self.ids, device=device, host=self)
    def tensor(ids, *, dtype, device, pin_memory=False):
        events.append(("tensor", tuple(ids), dtype, device, pin_memory))
        if device != "cpu":
            return NS(ids=tuple(ids), device=device)
        return HostIDs(ids)
    torch.tensor = tensor
    kernels = types.ModuleType(f"{PACKAGE}.metadata_reset")
    kernels.reset_metadata = lambda ids, scratch, pages: events.append(
        ("reset", ids, scratch, pages))
    package = types.ModuleType(PACKAGE)
    package.metadata_reset = kernels
    return {"torch": torch, PACKAGE: package, kernels.__name__: kernels}


def view_key(pointer, pages=12, device="cuda:0"):
    return (pointer, (pages, 16, 128, 128), (262144, 16384, 128, 1), device, "f32")


class TwoBOrderedGDNContracts(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_package_import_is_cpu_only_and_has_no_independent_hook(self):
        source = GDN.parents[3]
        program = (f"import sys; sys.path.insert(0, {str(source)!r}); "
                   "import vllm_mach.mxfp8.two_b.gdn as gdn; "
                   "assert not hasattr(gdn, 'install_worker_hook'); "
                   "assert not any(x in sys.modules for x in ('torch','vllm','triton'))")
        subprocess.run([sys.executable, "-c", program], check=True)
        self.assertEqual({p.name for p in GDN.glob("*.py")}, {
            "__init__.py", "dispatch.py", "worker.py", "ordered_allm_triton.py",
            "metadata_reset.py", "mixed.py", "mixed_ordered_triton.py"})

    def test_frozen_kernels_and_installed_source_guards(self):
        pins = {"ordered_allm_triton.py": "44b817a3cb97ef03c303c63d5c08e80c9206e3ab72cedc7425dc8f8d76f3a69d",
                "metadata_reset.py": "f60276ba7b0a61e6ae889658bcfca7ba7dc28d7809778b009c3565a541372226",
                "dispatch.py": "2edf5ce6370cc22cbe6c894df3e29abcb3604e31b7169608ee4bdab34e31220d"}
        for name, expected in pins.items():
            self.assertEqual(hashlib.sha256((GDN / name).read_bytes()).hexdigest(), expected, name)
        worker = load_worker()
        manifest = json.loads((GDN.parents[1] / "data/runtime_sources.json").read_text())
        for value, path in ((worker.GDN_SHA, "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"),
                            (worker.RUNNER_SHA, "v1/worker/gpu/model_runner.py"),
                            (worker.FLA_SHA, "third_party/flash_linear_attention/ops/fused_recurrent.py")):
            self.assertEqual(value, manifest["files"][f"vllm/{path}"]["installed_sha256"])
        self.assertEqual(worker.ROWS, (32, 48, 64, 96, 128, 160))
        self.assertEqual((worker.STATE_HEADS, worker.QKV_WIDTH, worker.QKV_STRIDE, worker.W),
                         (16, 6144, 8192, 4))

    def test_recurrent_math_matches_qualified_fp32_and_original_bf16(self):
        import copy
        fixture = json.loads((Path(__file__).parent / "fixtures/mxfp8_2b_recurrent_order.json").read_text())
        tree = ast.parse((GDN / "ordered_allm_triton.py").read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ordered_decode")
        original = ast.parse(fixture["original_kernel"]).body[0]

        class Select(ast.NodeTransformer):
            def __init__(self, enabled): self.enabled = enabled
            def visit_AnnAssign(self, node):
                if isinstance(node.target, ast.Name) and node.target.id == "legacy_fp32":
                    return None
                return self.generic_visit(node)
            def visit_If(self, node):
                if isinstance(node.test, ast.Name) and node.test.id == "legacy_fp32":
                    return [self.visit(n) for n in (node.body if self.enabled else node.orelse)]
                return self.generic_visit(node)

        replacements = fixture["qualified_replacements"]
        class Qualified(ast.NodeTransformer):
            def visit_Call(self, node):
                key = ast.unparse(node)
                if key in replacements:
                    return ast.parse(replacements[key], mode="eval").body
                return self.generic_visit(node)

        self.assertEqual(ast.dump(Select(False).visit(copy.deepcopy(kernel))), ast.dump(original))
        self.assertEqual(ast.dump(Select(True).visit(copy.deepcopy(kernel))),
                         ast.dump(Qualified().visit(copy.deepcopy(original))))
        decode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
        # The actual dtype/row dispatch, Gluon delegation, launch ABI/config,
        # state lifecycle and all unsupported-input checks are unchanged.
        self.assertEqual(ast.dump(decode), ast.dump(ast.parse(fixture["original_decode"]).body[0]))

    def test_recurrent_local4_gate_excludes_bf16_and_other_geometry(self):
        tree = ast.parse((GDN / "ordered_allm_triton.py").read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ordered_decode")
        gate = next(n.value for n in kernel.body if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == "legacy_fp32")
        code = compile(ast.Expression(gate), "local4_geometry", "eval")
        def enabled(dtype="fp32", **changes):
            values = dict(H=16, HV=16, K=128, V=128, BV=32, NV=4, W=4)
            values.update(changes)
            return eval(code, {"__builtins__": {}}, dict(values,
                base=NS(dtype=NS(element_ty=dtype)), tl=NS(float32="fp32")))
        self.assertTrue(enabled())
        self.assertFalse(enabled("bf16"))
        self.assertFalse(enabled("fp16"))
        for name, value in (("H", 8), ("HV", 32), ("K", 64), ("V", 64),
                            ("BV", 64), ("NV", 2), ("W", 8)):
            with self.subTest(field=name): self.assertFalse(enabled(**{name: value}))
        dispatch = ast.parse((GDN / "dispatch.py").read_text())
        fn = next(n for n in dispatch.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
        self.assertEqual(ast.unparse(fn.body[0]), "return high.decode(*args, **kwargs)")

    def test_shared_recurrent_helper_is_qualified_and_in_source_receipt(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/mxfp8_2b_recurrent_order.json").read_text())
        helper = GDN.parents[1] / "gdn/legacy_reduction.py"
        functions = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                     for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(functions, fixture["qualified_helper_functions"])
        worker = load_worker()
        self.assertEqual(worker._sha(helper), hashlib.sha256(helper.read_bytes()).hexdigest())
        source = (GDN / "worker.py").read_text()
        self.assertIn('sources["../../gdn/legacy_reduction.py"] = _sha(helper)', source)
        self.assertIn('"legacy_local4_fp32" if selected_dtype() == "float32"', source)

    def test_only_frozen_2b_bf16_kv_and_profile_configuration(self):
        worker = load_worker()
        torch = NS(bfloat16="bf16")
        self.assertTrue(worker.PINNED_IDS)
        with patch.dict(sys.modules, {"torch": torch}):
            for dtype in ("auto", "bfloat16"):
                worker._check_config(config(dtype))
            for dtype in ("fp8_e4m3", "fp8_e5m2", "float16"):
                with self.assertRaises(RuntimeError): worker._check_config(config(dtype))
            for section, field, value in (
                ("parallel_config", "tensor_parallel_size", 2),
                ("cache_config", "enable_prefix_caching", True),
                ("cache_config", "use_replayssm", True),
                ("cache_config", "mamba_cache_mode", "all"),
                ("cache_config", "kv_offloading_size", 1),
                ("cache_config", "kv_cache_memory_bytes", 4*2**30),
                ("model_config", "cpu_offload_gb", 1),
                ("model_config", "max_model_len", 8192),
                ("scheduler_config", "max_num_seqs", 128)):
                candidate = config()
                setattr(getattr(candidate.vllm_config, section), field, value)
                with self.assertRaises(RuntimeError): worker._check_config(candidate)
            candidate = config()
            candidate.vllm_config.model_config.hf_text_config.linear_num_value_heads = 32
            with self.assertRaisesRegex(RuntimeError, "2B model dimensions"):
                worker._check_config(candidate)
            for field in ("speculative_config", "kv_transfer_config"):
                candidate = config()
                setattr(candidate.vllm_config, field, NS())
                with self.assertRaises(RuntimeError): worker._check_config(candidate)
            with patch.dict(os.environ, {"VLLM_MACH_2B_PINNED_IDS": "0"}):
                with self.assertRaisesRegex(RuntimeError, "policy changed"):
                    worker._check_config(config())

    def test_quality_rows_and_policy_are_immutable(self):
        worker = load_worker()
        with patch.dict(sys.modules, {"torch": NS(bfloat16="bf16")}):
            for row in (4, 32, 64):
                with patch.dict(os.environ, {"VLLM_MACH_MXFP8_MODE": "quality",
                                             "VLLM_MACH_MXFP8_QUALITY_ROWS": str(row)}):
                    candidate = config()
                    candidate.vllm_config.cache_config.kv_cache_memory_bytes = 4*2**30
                    candidate.vllm_config.model_config.max_model_len = 1024
                    candidate.vllm_config.scheduler_config.max_num_seqs = row
                    candidate.vllm_config.scheduler_config.max_num_batched_tokens = max(8192, row*256)
                    worker._check_config(candidate)
                    self.assertEqual(worker._mode_contract(), ("quality", row, 4*2**30))
                    candidate.vllm_config.scheduler_config.max_num_batched_tokens = 2048
                    with self.assertRaisesRegex(RuntimeError, "maxbatch"):
                        worker._check_config(candidate)
            for env in ({"VLLM_MACH_MXFP8_MODE": "quality"},
                        {"VLLM_MACH_MXFP8_QUALITY_ROWS": "32"},
                        {"VLLM_MACH_MXFP8_MODE": "unknown"},
                        {"VLLM_MACH_MXFP8_MODE": "quality", "VLLM_MACH_MXFP8_QUALITY_ROWS": "16"}):
                with patch.dict(os.environ, env):
                    with self.assertRaises(RuntimeError): worker._check_config(config())
            worker._RUN_MODE, worker._QUALITY_ROW = "quality", 32
            with self.assertRaisesRegex(RuntimeError, "mode changed"):
                worker._check_config(config())

    def test_fresh_pinned_h2d_precedes_resets_and_deduplicates_per_device(self):
        worker = load_worker()
        worker._READY = True
        worker._VIEW_SCRATCH = {view_key(1): "one", view_key(2): "two",
                                view_key(3, device="cuda:1"): "three"}
        events = []
        with patch.dict(sys.modules, reset_api(events)):
            worker._reset_allocated([4, 2, 4, 2], "new")
            self.assertEqual([e[0] for e in events],
                             ["tensor", "h2d", "reset", "reset", "tensor", "h2d", "reset"])
            self.assertEqual(events[0], ("tensor", (2, 4), "i32", "cpu", True))
            self.assertEqual(events[1][2:], ("cuda:0", True))
            self.assertIs(events[2][1], events[3][1])
            self.assertIsNot(events[2][1], events[6][1])
            first_host = events[1][1]
            worker._reset_allocated([3], "cached")
            self.assertIsNot(events[8][1], first_host)
            self.assertEqual(first_host.ids, (2, 4))
        self.assertEqual(worker._ALLOCATION_RESETS, {"new": 1, "cached": 1, "ids": 3})
        worker._VIEW_SCRATCH = {view_key(1, pages=4): "one"}
        events.clear()
        with patch.dict(sys.modules, reset_api(events)):
            with self.assertRaisesRegex(RuntimeError, "outside GDN state pool"):
                worker._reset_allocated([4], "new")
        self.assertEqual(events, [])

    def test_runner_reset_follows_assignment_and_rejects_retained_or_copied_state(self):
        worker = load_worker()
        worker._READY = True
        worker._VIEW_SCRATCH = {view_key(1): "one"}
        events = []
        class Runner:
            def add_requests(self, output): events.append(("add",)); return "added"
            def update_requests(self, output): events.append(("update",)); return "updated"
        api = types.ModuleType("vllm.v1.worker.gpu.model_runner")
        api.GPUModelRunner = Runner
        modules = reset_api(events)
        modules[api.__name__] = api
        with patch.dict(sys.modules, modules), patch.object(worker, "_sha", return_value=worker.RUNNER_SHA), \
                patch.object(worker.inspect, "getsourcefile", return_value="runner.py"):
            worker._install_runner_lifecycle()
            worker._install_runner_lifecycle()
            runner = Runner()
            output = NS(scheduled_new_reqs=[NS(num_computed_tokens=0, block_ids=([0, 3, 3], [2]))],
                        scheduled_cached_reqs=NS(new_block_ids=[([4, 4],)]), kv_cache_block_copies=[])
            self.assertEqual(runner.add_requests(output), "added")
            self.assertEqual([e[0] for e in events], ["add", "tensor", "h2d", "reset"])
            events.clear()
            self.assertEqual(runner.update_requests(output), "updated")
            self.assertEqual([e[0] for e in events], ["update", "tensor", "h2d", "reset"])
            events.clear()
            output.scheduled_new_reqs[0].num_computed_tokens = 1
            with self.assertRaisesRegex(RuntimeError, "retained state"):
                runner.add_requests(output)
            output.kv_cache_block_copies = [object()]
            with self.assertRaisesRegex(RuntimeError, "block copies"):
                runner.update_requests(output)
            self.assertEqual(events, [])

    def test_owner_pool_binding_and_fp32_state_rejections(self):
        worker = load_worker()
        api = owner_api()
        with patch.dict(sys.modules, {api.__name__: api}):
            owner = object()
            worker._claim_worker(owner)
            worker._claim_worker(api.WorkerWrapperBase(owner))
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                worker._claim_worker(object())
            with self.assertRaisesRegex(RuntimeError, "no initialized worker"):
                worker._claim_worker(api.WorkerWrapperBase(None))
            with self.assertRaisesRegex(RuntimeError, "initialize"):
                worker.prepare_pools(owner)
            with self.assertRaisesRegex(RuntimeError, "single cache binding"):
                worker.initialize_cache(owner)
        state = NS(dtype="f32", shape=(12, 16, 128, 128),
                   stride=lambda dim=None: (262144, 16384, 128, 1) if dim is None else (262144, 16384, 128, 1)[dim])
        layer = NS(tp_size=1, gqa_interleaved_layout=False, enable_packed_recurrent_decode=True,
                   _is_sm120=True, num_k_heads=16, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                   kv_cache=[NS(dtype="bf16"), state], A_log=NS(dtype="f32"), dt_bias=NS(dtype="bf16"))
        with patch.dict(sys.modules, {"torch": NS(float32="f32", bfloat16="bf16")}):
            self.assertTrue(worker._layout(layer, state))
            state.dtype = "f16"
            self.assertFalse(worker._layout(layer, state))
            state.dtype = "f32"
            layer.kv_cache[0].dtype = "fp8"
            self.assertFalse(worker._layout(layer, state))
        for invalid in ([[]], ((),), ([True],), (["1"],)):
            with self.assertRaises(TypeError): worker._flatten_new(invalid)
        self.assertEqual(worker._flatten_new(([0, 2, -1], [3, 2])), [2, 3, 2])
        self.assertEqual(worker._flatten_new(None), [])

    def test_explicit_materialize_reset_preserves_storage_and_rejects_rebinding(self):
        worker = load_worker()
        worker._READY = True
        events = []
        api = owner_api()
        gdn_api = types.ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
        class Layer:
            def __init__(self, state): self.kv_cache = [None, state]
        gdn_api.QwenGatedDeltaNetAttention = Layer
        class State:
            shape, device, dtype = (12, 16, 128, 128), "cuda:0", "f32"
            pointer = 1
            def data_ptr(self): return self.pointer
            def stride(self): return (262144, 16384, 128, 1)
        state = State()
        layers = [Layer(state), Layer(state)]
        owner = NS(device="cuda:0", get_model=lambda: NS(named_modules=lambda: enumerate(layers)),
                   model_runner=NS(req_states=NS(req_id_to_index={})))
        scratch = object()
        worker._VIEW_SCRATCH = {worker._view_key(state): scratch}
        component = NS(
            materialize_slots=lambda base, ids, live: events.append(("materialize", base, ids, live)),
            reset_slots=lambda base, ids, live, **kwargs: events.append(("reset", base, ids, live, kwargs)))
        worker._COMPONENT = component
        torch = NS(int32="i32", arange=lambda start, end, **kwargs: tuple(range(start, end)),
                   cuda=NS(is_current_stream_capturing=lambda: False,
                           synchronize=lambda device: events.append(("sync", device))))
        modules = {"torch": torch, api.__name__: api, gdn_api.__name__: gdn_api}
        with patch.dict(sys.modules, modules):
            self.assertEqual(worker.materialize_state(owner),
                             {"materialized_views": 1, "synchronized": True})
            self.assertEqual([e[0] for e in events], ["sync", "materialize", "sync"])
            self.assertIs(events[1][3], scratch)
            events.clear()
            result = worker.reset_state(owner, zero_base=True)
            self.assertTrue(result["scratch_storage_preserved"])
            self.assertEqual([e[0] for e in events], ["sync", "reset", "sync"])
            self.assertEqual(events[1][4], {"zero_base": True})
            self.assertIs(worker._VIEW_SCRATCH[worker._view_key(state)], scratch)
            with self.assertRaises(TypeError): worker.reset_state(owner, zero_base=1)
            state.pointer = 2
            with self.assertRaisesRegex(RuntimeError, "storage changed"):
                worker.materialize_state(owner)
            owner.model_runner.req_states.req_id_to_index["live"] = 0
            with self.assertRaisesRegex(RuntimeError, "drain requests"):
                worker.reset_state(owner)

    def test_stock_route_materializes_before_parent_core(self):
        # Stock core sees dense FP32 state after ordered terms have been flushed.
        tree = ast.parse((GDN / "worker.py").read_text())
        register = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_register_gdn")
        core = next(n for n in register.body if isinstance(n, ast.FunctionDef) and n.name == "core")
        statements = [ast.unparse(n) for n in core.body]
        materialize = next(i for i, text in enumerate(statements) if "component.materialize_slots" in text)
        self.assertIn("return _ORIG_CORE", statements[materialize + 1])


if __name__ == "__main__":
    unittest.main()
