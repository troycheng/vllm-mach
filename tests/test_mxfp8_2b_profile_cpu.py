"""CPU-only 2B profile routing, environment, and composed worker contracts."""
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import test_mxfp8_2b_ba_graph_cpu as graph_tests

SOURCE = Path(__file__).resolve().parents[1] / "src"
NS = types.SimpleNamespace
NAME = "qwen35-2b-mxfp8-champion-v1"
PRODUCTION_CAPTURES = [4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 2048]


def load(name):
    with patch.object(sys, "path", [str(SOURCE), *sys.path]):
        return importlib.import_module("vllm_mach." + name)


def worker_api(events):
    class Worker:
        def init_device(self): events.append("device"); return "device"
        def load_model(self): events.append("load"); return "load"
        def initialize_from_config(self): events.append("cache"); return "cache"
        def compile_or_warm_up_model(self): events.append("compile"); return "compile"
        def execute_model(self): events.append("execute"); return "execute"
    class Wrapper:
        def __init__(self, worker): self.worker = worker
    return Worker, {
        "vllm.v1.worker.gpu_worker": NS(Worker=Worker),
        "vllm.v1.worker.worker_base": NS(WorkerWrapperBase=Wrapper)}


class TwoBProfileContracts(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"VLLM_MACH_PROFILE": NAME, "VLLM_FLASH_ATTN_VERSION": "2"}, clear=True)
        env.start(); self.addCleanup(env.stop)
        self.profile = load("mxfp8.two_b.profile")
        self.serve = load("mxfp8.two_b.serve")

    def test_production_and_quality_geometry_match_explicit_cli(self):
        contract = self.profile.run_contract()
        self.assertEqual((contract["max_model_len"], contract["max_num_seqs"],
                          contract["max_num_batched_tokens"], contract["kv_cache_memory_bytes"]),
                         (16384, 160, 2048, 19*2**30))
        self.assertEqual(contract["compilation_config"]["cudagraph_capture_sizes"], PRODUCTION_CAPTURES)
        for row, captures in ((4, [4]), (32, [4, 32]), (64, [4, 32, 64])):
            with patch.dict(os.environ, {"VLLM_MACH_MXFP8_MODE": "quality", "VLLM_MACH_MXFP8_QUALITY_ROWS": str(row)}):
                contract = self.profile.run_contract()
                argv = self.serve.build_argv("/checkpoint", quality_rows=row)
                self.assertEqual(contract["max_model_len"], 1024)
                self.assertEqual(contract["max_num_seqs"], row)
                self.assertEqual(contract["max_num_batched_tokens"], max(8192, row*256))
                self.assertEqual(contract["kv_cache_memory_bytes"], 4*2**30)
                self.assertFalse(contract["production_throughput_profile"])
                self.assertEqual(contract["compilation_config"]["cudagraph_capture_sizes"], captures)
                self.assertEqual(json.loads(argv[argv.index("--compilation-config")+1]), contract["compilation_config"])
                self.assertEqual(argv[argv.index("--kv-cache-dtype")+1], "bfloat16")
                self.assertEqual(argv[argv.index("--attention-backend")+1], "FLASH_ATTN")
        for row in (0, 8, 16, 128):
            with self.assertRaises(ValueError): self.profile.quality_contract(row)
        for environment in ({"VLLM_MACH_MXFP8_MODE": "quality"},
                            {"VLLM_MACH_MXFP8_QUALITY_ROWS": "32"},
                            {"VLLM_MACH_MXFP8_MODE": "other"},
                            {"VLLM_MACH_2B_PINNED_IDS": "0"}):
            with patch.dict(os.environ, environment):
                with self.assertRaises(RuntimeError): self.profile.run_contract()

    def test_production_argv_ignores_inherited_quality_selection(self):
        for environment in ({"VLLM_MACH_MXFP8_MODE": "quality", "VLLM_MACH_MXFP8_QUALITY_ROWS": "64"},
                            {"VLLM_MACH_MXFP8_MODE": "quality"},
                            {"VLLM_MACH_MXFP8_QUALITY_ROWS": "4"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment):
                argv = self.serve.build_argv("/checkpoint")
                self.assertEqual(json.loads(argv[argv.index("--compilation-config")+1])["cudagraph_capture_sizes"],
                                 PRODUCTION_CAPTURES)
                self.assertEqual(argv[argv.index("--max-num-seqs")+1], "160")

    def test_launch_environment_clears_quality_and_uses_fresh_cache(self):
        manifest = load("mxfp8.install").load_manifest()
        with tempfile.TemporaryDirectory() as directory:
            os.environ.update({"VLLM_MACH_MXFP8_MODE": "quality", "VLLM_MACH_MXFP8_QUALITY_ROWS": "64",
                               "VLLM_MACH_NATIVE_MXFP8": "1", "VLLM_MACH_2B_PINNED_IDS": "0"})
            first = self.serve.configure_environment(directory)
            self.assertNotIn("VLLM_MACH_MXFP8_QUALITY_ROWS", os.environ)
            self.assertEqual(os.environ["VLLM_MACH_MXFP8_MODE"], "production")
            self.assertEqual(os.environ["VLLM_MACH_NATIVE_MXFP8"], "0")
            self.assertEqual(os.environ["VLLM_MACH_2B_PINNED_IDS"], "1")
            self.assertEqual(os.environ["VLLM_PLUGINS"], "mach")
            for mapping in ("required_disabled_legacy_environment", "required_disabled_compact_environment"):
                for key, value in manifest[mapping].items(): self.assertEqual(os.environ[key], value)
            second = self.serve.configure_environment(directory, quality_rows=4)
            self.assertEqual(os.environ["VLLM_MACH_MXFP8_QUALITY_ROWS"], "4")
            self.assertNotEqual(first["VLLM_CACHE_ROOT"], second["VLLM_CACHE_ROOT"])
            self.assertNotEqual(first["TRITON_CACHE_DIR"], second["TRITON_CACHE_DIR"])
            self.assertEqual(self.profile.run_contract()["mode"], "quality")

    def test_import_and_print_command_need_no_torch_and_do_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            program = f'''
import importlib.abc, json, sys
sys.path.insert(0, {str(SOURCE)!r})
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'vllm', 'triton'):
            raise AssertionError('GPU dependency imported: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
import vllm_mach.mxfp8.two_b.profile
import vllm_mach.plugin
from vllm_mach.mxfp8.two_b.serve import main
sys.argv = ['serve', '/checkpoint', '--run-dir', {directory!r}, '--print-command']
main()
assert not any(x in sys.modules for x in ('torch', 'vllm', 'triton'))
'''
            result = subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
            argv = json.loads(result.stdout)
            self.assertEqual(argv[:3], ["vllm", "serve", "/checkpoint"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def lifecycle(self, events):
        stack = ExitStack()
        self.addCleanup(stack.close)
        Worker, modules = worker_api(events)
        stack.enter_context(patch.dict(sys.modules, modules))
        profile = self.profile
        gdn, ba, graph = (load("mxfp8.two_b."+name) for name in ("gdn", "ba", "graph_policy"))
        native = load("mxfp8.worker")
        stack.enter_context(patch.object(profile, "check_environment", return_value={}))
        stack.enter_context(patch.object(profile, "check_runtime_sources", return_value={"pinned": True}))
        stack.enter_context(patch.object(profile, "_check_geometry", side_effect=lambda *args: events.append("geometry")))
        stack.enter_context(patch.object(profile, "write_receipt", side_effect=lambda obj, phase: events.append(phase)))
        stack.enter_context(patch.object(native, "install_worker_backend", side_effect=lambda obj: events.append("native")))
        stack.enter_context(patch.object(ba, "install", side_effect=lambda obj: events.append("ba")))
        stack.enter_context(patch.object(graph, "install_dispatch", side_effect=lambda: events.append("graph")))
        for name in ("install_runtime", "prepare_layers", "prepare_pools"):
            stack.enter_context(patch.object(gdn, name, side_effect=lambda obj, label=name: events.append(label)))
        state = load("mxfp8.two_b.gdn.worker")
        stack.enter_context(patch.object(state, "_READY", False))
        stack.enter_context(patch.object(state, "_CACHE_INITIALIZED", False))
        def initialized(obj):
            events.append("initialize_cache")
            state._CACHE_INITIALIZED = True
        stack.enter_context(patch.object(gdn, "initialize_cache", side_effect=initialized))
        return Worker, state

    def test_composed_lifecycle_and_repeat_cache_rejected_before_parent(self):
        events = []
        Worker, state = self.lifecycle(events)
        self.assertTrue(self.profile.install_worker_hook())
        self.assertFalse(self.profile.install_worker_hook())
        obj = Worker()
        self.assertEqual(obj.init_device(), "device")
        self.assertEqual(obj.load_model(), "load")
        self.assertEqual(obj.initialize_from_config(), "cache")
        before = list(events)
        with self.assertRaisesRegex(RuntimeError, "reallocation"):
            obj.initialize_from_config()
        self.assertEqual(events, before)
        self.assertEqual(obj.compile_or_warm_up_model(), "compile")
        self.assertEqual(obj.execute_model(), "execute")
        obj.execute_model()
        self.assertEqual(events, ["graph", "device", "geometry", "native", "install_runtime", "ba", "load",
                                  "prepare_layers", "cache", "initialize_cache", "prepare_pools", "compile", "compiled",
                                  "execute", "serving", "execute"])
        state._CACHE_INITIALIZED = False
        state._READY = True
        before = list(events)
        with self.assertRaisesRegex(RuntimeError, "reallocation"):
            obj.initialize_from_config()
        self.assertEqual(events, before)

    def test_quality_skips_production_dispatch_and_contract_drift_rejects_compile(self):
        for row in (4, 32, 64):
            events = []
            with self.subTest(row=row), patch.dict(os.environ, {
                "VLLM_MACH_MXFP8_MODE": "quality", "VLLM_MACH_MXFP8_QUALITY_ROWS": str(row)}):
                Worker, _ = self.lifecycle(events)
                self.profile.install_worker_hook()
                self.assertNotIn("graph", events)
                obj = Worker(); obj.init_device()
                with patch.dict(os.environ, {"VLLM_MACH_MXFP8_QUALITY_ROWS": "4" if row != 4 else "32"}):
                    before = list(events)
                    with self.assertRaisesRegex(RuntimeError, "before compilation"):
                        obj.compile_or_warm_up_model()
                    self.assertEqual(events, before)
                    with self.assertRaisesRegex(RuntimeError, "after registration"):
                        self.profile.install_worker_hook()

    def test_2b_rejects_existing_native_or_4b_profile(self):
        for marker in ("_mach_mxfp8_native_hook", "_mach_mxfp8_champion_hook"):
            events = []
            Worker, _ = self.lifecycle(events)
            setattr(Worker, marker, True)
            before = Worker.init_device
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                self.profile.install_worker_hook()
            self.assertIs(Worker.init_device, before)
            self.assertEqual(events, [])

    def test_native_rejects_an_existing_2b_profile(self):
        native = load("mxfp8.worker")
        Worker, modules = worker_api([])
        Worker._mach_2b_hook = True
        before = Worker.init_device
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "profile|hook|2B|2b"):
                native.install_worker_hook()
        self.assertIs(Worker.init_device, before)

    def test_4b_rejects_an_existing_2b_profile(self):
        profile = load("mxfp8.profile")
        choices, graph = load("mxfp8.compile_choices"), load("mxfp8.graph_policy")
        Worker, modules = worker_api([])
        Worker._mach_2b_hook = True
        before = Worker.init_device
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"VLLM_MACH_PROFILE": profile.NAME}), \
                patch.object(profile, "check_environment", return_value={}), \
                patch.object(profile, "check_runtime_sources", return_value={}), \
                patch.object(choices, "install", return_value=True), patch.object(graph, "install_dispatch", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "profile|hook|2B|2b"):
                profile.install_worker_hook()
        self.assertIs(Worker.init_device, before)

    def test_plugin_routes_2b_without_registering_other_backends(self):
        plugin = load("plugin")
        with patch.object(plugin, "version", return_value="0.29.0"), \
                patch.object(plugin, "register_dense_kernel") as dense, \
                patch.object(self.profile, "install_worker_hook") as hook:
            plugin.register()
            hook.assert_called_once_with()
            dense.assert_not_called()
        with patch.object(plugin, "version", return_value="0.29.0"), \
                patch.dict(os.environ, {"VLLM_MACH_PROFILE": "typo"}):
            with self.assertRaisesRegex(RuntimeError, "Unknown"):
                plugin.register()

    def test_inspection_requires_actual_graphs_and_all_eligible_gdn_captures(self):
        profile = self.profile
        graph, gdn, ba = (load("mxfp8.two_b."+name) for name in ("graph_policy", "gdn", "ba"))
        native = load("mxfp8.worker")
        for row in (None, 4, 32, 64):
            environment = {} if row is None else {
                "VLLM_MACH_MXFP8_MODE": "quality", "VLLM_MACH_MXFP8_QUALITY_ROWS": str(row)}
            with self.subTest(row=row), patch.dict(os.environ, environment):
                modules, Manager, Wrapper, sentinel = graph_tests.graph_api()
                Worker, owner_modules = worker_api([])
                modules.update(owner_modules)
                modules["torch"] = NS(bfloat16="bf16")
                worker = graph_tests.worker_fixture(Manager, Wrapper, sentinel, rows=row)
                cfg = worker.vllm_config
                cfg.model_config.hf_text_config = NS(hidden_size=2048, intermediate_size=6144,
                    num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=2, head_dim=256)
                cfg.attention_config = NS(backend=NS(name="FLASH_ATTN"))
                cfg.cache_config.cache_dtype = "bfloat16"
                worker._mach_2b_contract = profile.run_contract()
                worker._mach_2b_runtime_sources = {}
                worker.get_model = lambda: NS(named_modules=lambda: [("lm_head", NS(weight=NS(dtype="bf16", shape=(248320, 2048))))])
                layers = [f"layer{i}" for i in range(18)]
                eligible = [m for m in (32, 48, 64, 96, 128, 160) if m <= (row or 160)]
                state = {"ready": True, "layer_count": 18, "pinned_ids_enabled": True,
                         "layers": layers, "eligible_rows": eligible,
                         "capture_construction": {f"{name}|m{m}": 1 for name in layers for m in eligible}}
                with patch.dict(sys.modules, modules), patch.object(gdn, "inspect_worker", return_value=state), \
                        patch.object(ba, "verify_capture", return_value={}) as ba_verify, \
                        patch.object(native, "verify_worker_execution", return_value={}) as native_verify, \
                        patch.object(profile, "check_environment", return_value={}), \
                        patch.object(graph, "_INSTALLED", False):
                    if row is None:
                        graph.install_dispatch()
                    receipt = profile.inspect_worker(worker)
                    self.assertTrue(receipt["graph"]["ready"])
                    native_verify.assert_called_once_with(worker, require_capture=True)
                    ba_verify.assert_called_once_with(worker, required_rows=(4, 8) if row is None else (4,))
                    graph_entry = next(iter(worker.model_runner.cudagraph_manager.graphs))
                    worker.model_runner.cudagraph_manager.graphs[graph_entry] = None
                    with self.assertRaisesRegex(RuntimeError, "capture incomplete"):
                        profile.inspect_worker(worker)
                    worker.model_runner.cudagraph_manager.graphs[graph_entry] = object()
                    if eligible:
                        state["capture_construction"][f"layer0|m{row or 160}"] = 0
                        with self.assertRaisesRegex(RuntimeError, "Missing 2B GDN"):
                            profile.inspect_worker(worker)


if __name__ == "__main__":
    unittest.main()
