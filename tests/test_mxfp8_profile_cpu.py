"""Cross-module profile contracts and the live worker warmup call sequence.

Real Mach modules are imported together. CUDA entrypoints are replaced only
for lifecycle ordering checks; no torch/vLLM installation is needed.
"""
from collections import namedtuple
from contextlib import ExitStack
from enum import Enum
import importlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src"
PREFIXES = ("vllm", "torch", "mxfp6")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: v for k, v in sys.modules.items() if k.startswith(PREFIXES)}
        for key in self.saved:
            sys.modules.pop(key)
        sys.path.insert(0, str(SOURCE))
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.profile = importlib.import_module("vllm_mach.mxfp8.profile")
        self.serve = importlib.import_module("vllm_mach.mxfp8.serve")

    def tearDown(self):
        self.env.stop()
        sys.path.remove(str(SOURCE))
        for key in list(sys.modules):
            if key.startswith(PREFIXES):
                sys.modules.pop(key)
        sys.modules.update(self.saved)

    def modules(self):
        # Tensor annotations are deferred; importing these modules must not
        # access CUDA or load native libraries even when torch is present.
        sys.modules["torch"] = types.ModuleType("torch")
        return {name: importlib.import_module(f"vllm_mach.mxfp8.{name}") for name in
                ("worker", "native_backend", "dual", "ba", "gdn", "head", "kv",
                 "graph_policy", "projection_parallel", "compile_choices")}

    def test_parent_import_and_command_contract_are_cpu_only(self):
        importlib.import_module("vllm_mach.plugin")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm.v1.worker.gpu_worker", sys.modules)
        argv = self.serve.build_argv("/public/model")
        self.assertEqual(argv[argv.index("--kv-cache-memory") + 1], str(19 * 2**30))
        config = json.loads(argv[argv.index("--compilation-config") + 1])
        self.assertIn(2048, config["cudagraph_capture_sizes"])
        self.assertEqual(config["max_cudagraph_capture_size"], 2048)
        for rows in (32, 64):
            argv = self.serve.build_argv("/public/model", quality_rows=rows)
            self.assertEqual(argv[argv.index("--kv-cache-memory") + 1], str((4 if rows == 32 else 19) * 2**30))
            self.assertEqual(argv[argv.index("--max-num-seqs") + 1], str(rows))
            self.assertEqual(argv[argv.index("--max-model-len") + 1], "1024")
            self.assertEqual(argv[argv.index("--max-num-batched-tokens") + 1], str(rows * 256))
            self.assertIn("--disable-log-stats", argv)
            self.assertEqual(argv[argv.index("--logprobs-mode") + 1], "raw_logprobs")
            config = json.loads(argv[argv.index("--compilation-config") + 1])
            self.assertEqual(config, {"cudagraph_capture_sizes": [4, 32] if rows == 32 else [4, 32, 64],
                                     "max_cudagraph_capture_size": rows,
                                     "cudagraph_mode": "FULL_AND_PIECEWISE"})
        self.assertNotIn("torch", sys.modules)

    def test_real_component_signatures_accept_lifecycle_calls(self):
        modules = self.modules()
        worker = object()
        calls = {
            "worker": [("install_worker_backend", (worker,), {}),
                       ("verify_worker_execution", (worker,), {"require_capture": True})],
            "native_backend": [("inspect", (worker,), {})],
            "dual": [("install", (worker,), {}), ("prepare_model", (worker,), {}),
                     ("verify_capture", (worker,), {}), ("inspect_worker", (worker,), {})],
            "ba": [("install", (worker,), {}), ("prepare_model", (worker,), {}),
                   ("verify_capture", (worker,), {}), ("inspect_worker", (worker,), {})],
            "gdn": [(name, (worker,), {}) for name in
                    ("install_runtime", "prepare_layers", "initialize_cache", "prepare_pools", "inspect_worker")],
            "head": [("install_head", (worker,), {}), ("inspect_head", (worker,), {})],
            "kv": [("install_scales", (worker,), {}), ("inspect_scales", (worker,), {})],
            "graph_policy": [("install_dispatch", (), {}), ("verify_capture", (worker,), {})],
            "projection_parallel": [("prepare_model", (worker,), {"cache_directory": Path("/public/aot")}),
                                    ("verify_capture", (worker,), {}), ("inspect_worker", (worker,), {})],
            "compile_choices": [("install", (), {}), ("verify_coverage", (), {}), ("inspect", (), {})],
        }
        for component, entries in calls.items():
            for name, args, kwargs in entries:
                with self.subTest(component=component, name=name):
                    inspect.signature(getattr(modules[component], name)).bind(*args, **kwargs)
        self.assertEqual(len(modules["compile_choices"].load_policy()["bindings"]), 23)

    def test_environment_resets_stale_quality_and_sets_compile_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            startup_directories = set()
            for rows in (None, None, 32, 64, None):
                os.environ.update(VLLM_USE_AOT_COMPILE="0", VLLM_DISABLE_COMPILE_CACHE="1",
                                  TORCH_COMPILE_FORCE_DISABLE_CACHES="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
                                  VLLM_CACHE_ROOT="/stale/vllm", TORCHINDUCTOR_CACHE_DIR="/stale/inductor",
                                  TRITON_CACHE_DIR="/stale/triton")
                settings = self.serve.configure_environment(directory, quality_rows=rows)
                self.assertEqual(self.profile.run_contract()["quality_rows"], rows)
                self.assertEqual(os.environ["VLLM_USE_AOT_COMPILE"], "1")
                self.assertEqual(os.environ["VLLM_DISABLE_COMPILE_CACHE"], "0")
                self.assertEqual(settings["VLLM_USE_AOT_COMPILE"], "1")
                self.assertEqual(settings["VLLM_DISABLE_COMPILE_CACHE"], "0")
                for name in ("TORCH_COMPILE_FORCE_DISABLE_CACHES", "TORCHINDUCTOR_FORCE_DISABLE_CACHES"):
                    self.assertEqual(os.environ[name], "0")
                    self.assertEqual(settings[name], "0")
                cache_paths = [Path(settings[name]) for name in
                               ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")]
                startup = cache_paths[0].parent
                self.assertTrue(all(path.parent == startup for path in cache_paths))
                self.assertEqual(startup.parent, Path(directory).resolve() / "cache")
                self.assertTrue(startup.name.startswith("startup-"))
                self.assertTrue(startup.is_dir())
                self.assertNotIn(startup, startup_directories)
                startup_directories.add(startup)
                (startup / "retained-cache").write_text("previous startup")
                for previous in startup_directories:
                    self.assertTrue((previous / "retained-cache").is_file())
                for name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                    self.assertEqual(os.environ[name], settings[name])
            manifest = importlib.import_module("vllm_mach.mxfp8.install").load_manifest()
            for name, value in {**manifest["required_disabled_legacy_environment"],
                                **manifest["required_disabled_compact_environment"]}.items():
                self.assertEqual(os.environ[name], value)
            self.assertNotIn("VLLM_MACH_MXFP8_QUALITY_ROWS", os.environ)
            self.assertTrue(self.profile.run_contract()["production_throughput_profile"])
            validate = types.ModuleType("vllm_mach.mxfp8.prepare_model")
            def validate_model(*args, **kwargs):
                self.assertEqual(os.environ["TORCHINDUCTOR_COMPILE_THREADS"], "1")
                self.assertEqual(os.environ["VLLM_USE_AOT_COMPILE"], "1")
                self.assertEqual(os.environ["VLLM_DISABLE_COMPILE_CACHE"], "0")
                self.assertEqual(os.environ["TORCH_COMPILE_FORCE_DISABLE_CACHES"], "0")
                self.assertEqual(os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"], "0")
                self.assertEqual(os.environ["VLLM_MACH_MXFP8_MODE"], "quality")
                self.assertTrue(Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).is_relative_to(Path(directory).resolve()))
                return {"validated": True}
            validate.validate_model = validate_model
            sys.modules[validate.__name__] = validate
            cli = types.ModuleType("vllm.entrypoints.cli.main")
            cli.main = lambda: None
            sys.modules[cli.__name__] = cli
            os.environ.update(VLLM_USE_AOT_COMPILE="0", VLLM_DISABLE_COMPILE_CACHE="1",
                              TORCH_COMPILE_FORCE_DISABLE_CACHES="1", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
            with patch.object(self.profile, "check_environment", return_value={}), patch.object(
                    self.profile, "check_runtime_sources", return_value={"state": "installed"}), patch.object(
                    sys, "argv", ["serve", "/public/model", "--run-dir", directory, "--quality-rows", "32"]):
                self.serve.main()
            receipt = json.loads((Path(directory) / "launch.json").read_text())
            self.assertEqual(receipt["mode"], "quality")
            self.assertFalse(receipt["production_throughput_profile"])
            self.assertEqual(receipt["environment"]["VLLM_USE_AOT_COMPILE"], "1")
            self.assertEqual(receipt["environment"]["VLLM_DISABLE_COMPILE_CACHE"], "0")
            self.assertEqual(receipt["environment"]["TORCH_COMPILE_FORCE_DISABLE_CACHES"], "0")
            self.assertEqual(receipt["environment"]["TORCHINDUCTOR_FORCE_DISABLE_CACHES"], "0")
            for name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                self.assertEqual(receipt["environment"][name], os.environ[name])

    def test_parent_and_worker_gate_requires_installed_sources_and_disabled_paths(self):
        installer = importlib.import_module("vllm_mach.mxfp8.install")
        upstream, installed = b"# upstream\n", b"# formal runtime\n"
        digest = lambda value: hashlib.sha256(value).hexdigest()
        manifest = {"files": {"vllm/runtime.py": {"upstream_sha256": digest(upstream),
                    "installed_sha256": digest(installed)}},
                    "required_disabled_legacy_environment": {"VLLM_MACH_DENSE_AUTO": "0"},
                    "required_disabled_compact_environment": {"VLLM_HYBRID_NVFP4_LM_HEAD": "0"}}
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            source = site / "vllm/runtime.py"
            source.parent.mkdir()
            source.write_bytes(upstream)
            with patch.object(installer, "load_manifest", return_value=manifest), patch.object(
                    self.profile, "distribution", return_value=types.SimpleNamespace(locate_file=lambda path: site)):
                with self.assertRaisesRegex(RuntimeError, "Install the MXFP8 runtime"):
                    self.profile.check_runtime_sources()
                source.write_bytes(installed)
                with self.assertRaisesRegex(RuntimeError, "disabled legacy/compact"):
                    self.profile.check_runtime_sources()
                os.environ.update(VLLM_MACH_DENSE_AUTO="0", VLLM_HYBRID_NVFP4_LM_HEAD="0")
                receipt = self.profile.check_runtime_sources()
                self.assertEqual(receipt["sha256"]["vllm/runtime.py"], digest(installed))
                os.environ["VLLM_MACH_DENSE_AUTO"] = "1"
                with self.assertRaisesRegex(RuntimeError, "disabled legacy/compact"):
                    self.profile.check_runtime_sources()
                source.write_bytes(b"# foreign\n")
                with self.assertRaisesRegex(RuntimeError, "source SHA256 mismatch"):
                    self.profile.check_runtime_sources()
        self.assertNotIn("torch", sys.modules)

    def lifecycle(self, *, quality_rows=None, failure=False):
        modules = self.modules()
        events = []
        class Worker:
            def init_device(self): events.append("original_init")
            def load_model(self, *, load_dummy_weights=False): events.append("original_load")
            def initialize_from_config(self, config): events.append("original_cache")
            def execute_model(self, output):
                events.append("original_execute")
                return output
            def compile_or_warm_up_model(self):
                events.append("original_compile")
                # Pinned vLLM warmup_kernels calls this before compile returns.
                self.execute_model("warmup")
                return "compiled"
        gpu = types.ModuleType("vllm.v1.worker.gpu_worker")
        gpu.Worker = Worker
        sys.modules[gpu.__name__] = gpu
        os.environ["VLLM_MACH_PROFILE"] = self.profile.NAME
        os.environ["VLLM_MACH_MXFP8_RUN_DIR"] = tempfile.gettempdir()
        if quality_rows:
            os.environ["VLLM_MACH_MXFP8_MODE"] = "quality"
            os.environ["VLLM_MACH_MXFP8_QUALITY_ROWS"] = str(quality_rows)
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.profile, "check_environment", return_value={}))
            stack.enter_context(patch.object(self.profile, "check_runtime_sources", return_value={"state": "installed"}))
            stack.enter_context(patch.object(self.profile, "write_receipt", side_effect=lambda worker, phase: events.append(phase)))
            stack.enter_context(patch.object(self.profile, "_verify_quality_capture", side_effect=lambda *args: events.append("quality_capture")))
            targets = {"compile_choices": ["install", "verify_coverage"],
                       "graph_policy": ["install_dispatch", "verify_capture"],
                       "worker": ["install_worker_backend", "verify_worker_execution"],
                       "dual": ["install", "prepare_model", "verify_capture"],
                       "ba": ["install", "prepare_model", "verify_capture"],
                       "gdn": ["install_runtime", "prepare_layers", "initialize_cache", "prepare_pools"],
                       "kv": ["install_scales", "inspect_scales"],
                       "projection_parallel": ["prepare_model", "verify_capture"], "head": ["install_head"]}
            for component, names in targets.items():
                for name in names:
                    label = f"{component}.{name}"
                    def call(*args, _label=label, **kwargs):
                        events.append(_label)
                        if failure and _label == "head.install_head":
                            raise RuntimeError("head failed")
                    stack.enter_context(patch.object(modules[component], name, side_effect=call))
            self.assertTrue(self.profile.install_worker_hook())
            self.assertFalse(self.profile.install_worker_hook())
            worker = Worker()
            worker.init_device()
            worker.load_model(load_dummy_weights=False)
            worker.initialize_from_config(object())
            if failure:
                with self.assertRaisesRegex(RuntimeError, "head failed"):
                    worker.compile_or_warm_up_model()
                self.assertFalse(worker._mach_mxfp8_ready)
                self.assertNotIn("compiled", events)
                self.assertNotIn("serving", events)
            else:
                self.assertEqual(worker.compile_or_warm_up_model(), "compiled")
                self.assertEqual(worker._mach_mxfp8_steps, 0)
                self.assertNotIn("serving", events)
                self.assertTrue(worker._mach_mxfp8_ready)
                self.assertEqual(worker.execute_model("request"), "request")
                worker.execute_model("second")
                self.assertEqual(events.count("serving"), 1)
                self.assertEqual(worker._mach_mxfp8_steps, 2)
            os.environ["VLLM_MACH_MXFP8_MODE"] = "quality" if not quality_rows else "production"
            if quality_rows:
                os.environ.pop("VLLM_MACH_MXFP8_QUALITY_ROWS")
            else:
                os.environ["VLLM_MACH_MXFP8_QUALITY_ROWS"] = "32"
            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                self.profile.install_worker_hook()
        return events

    def test_compile_warmup_is_not_serving_and_lifecycle_order_is_preserved(self):
        events = self.lifecycle()
        expected = ["original_init", "worker.install_worker_backend", "dual.install", "ba.install",
                    "gdn.install_runtime", "original_load", "kv.install_scales", "dual.prepare_model",
                    "ba.prepare_model", "gdn.prepare_layers", "projection_parallel.prepare_model",
                    "original_cache", "gdn.initialize_cache", "gdn.prepare_pools", "original_compile",
                    "original_execute", "worker.verify_worker_execution", "dual.verify_capture", "ba.verify_capture",
                    "graph_policy.verify_capture", "projection_parallel.verify_capture", "compile_choices.verify_coverage",
                    "kv.inspect_scales", "head.install_head", "compiled"]
        start = events.index("original_init")
        self.assertEqual(events[start:start + len(expected)], expected)

    def test_failed_final_preparation_never_marks_worker_ready(self):
        self.lifecycle(failure=True)

    def test_quality_does_not_install_pw_or_claim_all_production_coverage(self):
        events = self.lifecycle(quality_rows=64)
        self.assertIn("quality_capture", events)
        for name in ("graph_policy.install_dispatch", "graph_policy.verify_capture", "dual.verify_capture",
                     "ba.verify_capture", "projection_parallel.verify_capture", "compile_choices.verify_coverage"):
            self.assertNotIn(name, events)

    def quality_fixture(self, rows):
        modules = self.modules()
        class Mode(Enum):
            NONE = 0
            PIECEWISE = 1
            FULL = 2
            FULL_AND_PIECEWISE = (2, 1)
        compilation = types.ModuleType("vllm.config.compilation")
        compilation.CUDAGraphMode = Mode
        sys.modules[compilation.__name__] = compilation
        captures = [4, 32] if rows == 32 else [4, 32, 64]
        Desc = namedtuple("Desc", "cg_mode num_tokens")
        full = [Desc(Mode.FULL, m) for m in captures]
        pw = [Desc(Mode.PIECEWISE, m) for m in captures]
        def dispatch(*, num_tokens, uniform_token_count, **kwargs):
            mode = Mode.FULL if uniform_token_count == 1 else Mode.PIECEWISE
            if num_tokens > rows:
                return Desc(Mode.NONE, num_tokens)
            return Desc(mode, next(m for m in captures if m >= num_tokens))
        manager = types.SimpleNamespace(_graphs_captured=True,
            _capture_descs={Mode.FULL: full, Mode.PIECEWISE: pw},
            graphs={d: object() for d in full}, cudagraph_mode=Mode.FULL_AND_PIECEWISE,
            use_breakable_cg=False, dispatch=dispatch)
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(cudagraph_manager=manager),
            vllm_config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(cudagraph_mode=Mode.FULL_AND_PIECEWISE,
                cudagraph_capture_sizes=captures, max_cudagraph_capture_size=rows),
                cache_config=types.SimpleNamespace(kv_cache_memory_bytes=(4 if rows == 32 else 19) * 2**30),
                model_config=types.SimpleNamespace(max_model_len=1024),
                scheduler_config=types.SimpleNamespace(max_num_seqs=rows, max_num_batched_tokens=rows * 256)))
        cuda_graph = types.ModuleType("vllm.compilation.cuda_graph")
        wrapper = types.SimpleNamespace(vllm_config=worker.vllm_config, runtime_mode=Mode.PIECEWISE,
            concrete_cudagraph_entries={d: types.SimpleNamespace(cudagraph=object()) for d in pw})
        cuda_graph.CUDAGraphWrapper = types.SimpleNamespace(_all_instances=[wrapper])
        sys.modules[cuda_graph.__name__] = cuda_graph
        state = {"profile_mode": "quality", "quality_rows": rows, "ready": True, "cache_initialized": True,
                 "layer_count": 24, "layers": [f"gdn{i}" for i in range(24)],
                 "capture_construction": {f"gdn{i}|m{m}": 1 for m in captures for i in range(24)}}
        dual = {"installed": True, "capture_layers": {str(m): {i: 1 for i in range(24)} for m in captures if m in (32, 64)},
                "mlp_capture_layers": {i: 1 for i in range(32)}}
        pair = {"ready": True, "aot_modules": ["public-generated-module"], "pairs": {str(i): {} for i in range(24)},
                "counts": [{"phase": "capture", "m": m, "layer_id": i, "parallel": True, "calls": 1}
                           for m in captures if m in (32, 64) for i in range(1, 24)]}
        return modules, Mode, worker, state, dual, pair

    def test_quality_checks_actual_selected_geometry_and_all_target_layers(self):
        modules, Mode, worker, state, dual, pair = self.quality_fixture(32)
        with ExitStack() as stack:
            for component, receipt in (("gdn", state), ("dual", dual), ("projection_parallel", pair)):
                stack.enter_context(patch.object(modules[component], "inspect_worker", return_value=receipt))
            ba = stack.enter_context(patch.object(modules["ba"], "verify_capture", return_value={}))
            choices = stack.enter_context(patch.object(modules["compile_choices"], "verify_coverage", return_value={
                "installed": True, "source_mismatches": [], "compile_threads": "1"}))
            self.assertFalse(self.profile._verify_quality_capture(worker, 32)["production_throughput_profile"])
            choices.assert_called_once_with(require_complete=True)
            ba.assert_called_once_with(worker, required_rows=(4,))
            choices.side_effect = RuntimeError("quality recipes missing")
            with self.assertRaisesRegex(RuntimeError, "quality recipes missing"):
                self.profile._verify_quality_capture(worker, 32)
            choices.side_effect = None
            worker.model_runner.cudagraph_manager._capture_descs[Mode.PIECEWISE].append(
                type(worker.model_runner.cudagraph_manager._capture_descs[Mode.PIECEWISE][0])(Mode.PIECEWISE, 2048))
            with self.assertRaisesRegex(RuntimeError, "FULL_AND_PIECEWISE"):
                self.profile._verify_quality_capture(worker, 32)
            worker.model_runner.cudagraph_manager._capture_descs[Mode.PIECEWISE].pop()
            pair["counts"].pop()
            with self.assertRaisesRegex(RuntimeError, "projection pair capture incomplete"):
                self.profile._verify_quality_capture(worker, 32)
            pair["counts"].append({"phase": "capture", "m": 32, "layer_id": 23, "parallel": True, "calls": 1})
            state["capture_construction"].pop("gdn23|m32")
            with self.assertRaisesRegex(RuntimeError, "GDN capture incomplete"):
                self.profile._verify_quality_capture(worker, 32)

    def test_quality_graph_plan_is_not_evidence_of_real_capture(self):
        _, Mode, worker, _, _, _ = self.quality_fixture(32)
        manager = worker.model_runner.cudagraph_manager
        graph = self.profile._quality_graph_receipt(worker, 32)
        self.assertEqual(graph["piecewise_sizes"], [4, 32])
        self.assertEqual(graph["piecewise_graph_counts"], {"4": 1, "32": 1})
        full_desc, full_graph = manager.graphs.popitem()
        with self.assertRaisesRegex(RuntimeError, "FULL graph capture incomplete"):
            self.profile._quality_graph_receipt(worker, 32)
        manager.graphs[full_desc] = full_graph
        wrappers = sys.modules["vllm.compilation.cuda_graph"].CUDAGraphWrapper._all_instances
        entry = next(iter(wrappers[0].concrete_cudagraph_entries.values()))
        entry.cudagraph = None
        with self.assertRaisesRegex(RuntimeError, "PIECEWISE graph capture incomplete"):
            self.profile._quality_graph_receipt(worker, 32)
        entry.cudagraph = object()
        wrappers[0].vllm_config = object()
        with self.assertRaisesRegex(RuntimeError, "PIECEWISE graph capture incomplete"):
            self.profile._quality_graph_receipt(worker, 32)

    def test_quality_checks_resolved_mode_and_dispatch_without_production_policy(self):
        _, Mode, worker, _, _, _ = self.quality_fixture(32)
        manager = worker.model_runner.cudagraph_manager
        manager.cudagraph_mode = Mode.FULL
        with self.assertRaisesRegex(RuntimeError, "FULL_AND_PIECEWISE"):
            self.profile._quality_graph_receipt(worker, 32)
        manager.cudagraph_mode = Mode.FULL_AND_PIECEWISE
        dispatch = manager.dispatch
        manager.dispatch = lambda **kw: types.SimpleNamespace(cg_mode=Mode.NONE, num_tokens=kw["num_tokens"])
        with self.assertRaisesRegex(RuntimeError, "dispatch changed"):
            self.profile._quality_graph_receipt(worker, 32)
        def padded_tail(**kw):
            if kw["num_tokens"] > 32:
                return types.SimpleNamespace(cg_mode=Mode.PIECEWISE, num_tokens=2048)
            return dispatch(**kw)
        manager.dispatch = padded_tail
        with self.assertRaisesRegex(RuntimeError, "uncaptured tail"):
            self.profile._quality_graph_receipt(worker, 32)

    def test_quality_m64_requires_m32_capture_and_accepted_19gib_geometry(self):
        modules, _, worker, state, dual, pair = self.quality_fixture(64)
        with ExitStack() as stack:
            for component, receipt in (("gdn", state), ("dual", dual), ("projection_parallel", pair)):
                stack.enter_context(patch.object(modules[component], "inspect_worker", return_value=receipt))
            stack.enter_context(patch.object(modules["ba"], "verify_capture", return_value={}))
            stack.enter_context(patch.object(modules["compile_choices"], "verify_coverage", return_value={}))
            self.assertEqual(self.profile._verify_quality_capture(worker, 64)["full_sizes"], [4, 32, 64])
            worker.vllm_config.cache_config.kv_cache_memory_bytes = 4 * 2**30
            with self.assertRaisesRegex(RuntimeError, "accepted capacity/scheduler"):
                self.profile._verify_quality_capture(worker, 64)
            worker.vllm_config.cache_config.kv_cache_memory_bytes = 19 * 2**30
            dual["capture_layers"]["32"].pop(23)
            with self.assertRaisesRegex(RuntimeError, "M32 dual QKVZ capture incomplete"):
                self.profile._verify_quality_capture(worker, 64)


if __name__ == "__main__":
    unittest.main()
