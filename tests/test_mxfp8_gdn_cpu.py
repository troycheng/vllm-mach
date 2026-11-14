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

if os.environ.get("MACH_TEST_RUNTIME_PROFILE"):
    from importlib.metadata import distribution
    GDN = Path(distribution("vllm-mach").locate_file("vllm_mach/mxfp8/gdn/worker.py")).resolve().parent
else:
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
        # The two BF16 changes are separately bound to the old-exact component
        # and original fallback ASTs by mxfp8_bf16_recurrent_order.json.
        fingerprints = {
            "lowm_allshape_triton": "17220ee86194044b5792a098e4f3c753992fdf4a83ac2c97600aebb603fabb94",
            "ordered_allm_triton": "dab59137c278acc641191ab5bd3570dc5011e523fbece166a5b9b38f0bd15315",
            "ordered_m4_deferred_triton": "85fafccf509727d73972e0ba9b077449587869b0df4931df2e22845887f34e13",
            "ordered_m16_deferred_triton": "add8b45c3a8cb2c70bea6a98db434bf66200016ccfd3cdc89514de14c2223c63",
            "bf16_small_gluon": "9eff8aec4f1bfd1d0aa9c3d2b5eb505442aec27dcfcf383905f6135f5490dfd7",
        }
        for name, expected in fingerprints.items():
            tree = ast.parse((GDN / f"{name}.py").read_text())
            functions = [ast.dump(n, include_attributes=False) for n in tree.body
                         if isinstance(n, ast.FunctionDef) and
                         (n.name.startswith("_ordered_") or n.name in ("_lowm_ordered_decode", "decode"))]
            actual = hashlib.sha256("\n".join(functions).encode()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_bf16_pdl_launch_and_wait_policy_agree(self):
        # No CUDA import: exercise the Python policy with a recording launcher.
        class Kernel:
            def __init__(self, fn):
                self.fn = fn

            def __getitem__(self, grid):
                return lambda *args, **kwargs: (args, kwargs)

        gluon = types.ModuleType("triton.experimental.gluon")
        language = types.ModuleType("triton.experimental.gluon.language")
        language.constexpr = object
        gluon.language, gluon.jit = language, Kernel
        experimental = types.ModuleType("triton.experimental")
        experimental.gluon = gluon
        helpers = types.ModuleType("vllm_mach.mxfp8.gdn.legacy_reduction_gluon")
        helpers._gl_old_square_128 = object()
        helpers._gl_old_dot_128_8local = object()
        modules = {"triton": types.ModuleType("triton"),
                   "triton.experimental": experimental,
                   "triton.experimental.gluon": gluon,
                   "triton.experimental.gluon.language": language,
                   "vllm_mach.mxfp8.gdn.legacy_reduction_gluon": helpers}
        tensor = types.SimpleNamespace(shape=(32, 32, 128, 128), stride=lambda dim: 1)
        scratch = types.SimpleNamespace(**{name: object() for name in
            ("pending_k", "pending_d", "coeff", "prefix", "age")})
        for setting, grouped in (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"),
                                 ("invalid", "0"), ("0", "invalid")):
            with patch.dict(sys.modules, modules), patch.dict(os.environ,
                    {"VLLM_MACH_MXFP8_BF16_GDN_PDL": setting,
                     "VLLM_MACH_MXFP8_BF16_GDN_GROUPED": grouped}):
                spec = importlib.util.spec_from_file_location(
                    "vllm_mach.mxfp8.gdn.bf16_pdl_policy", GDN / "bf16_small_gluon.py")
                module = importlib.util.module_from_spec(spec)
                if "invalid" in (setting, grouped):
                    with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                        spec.loader.exec_module(module)
                    continue
                spec.loader.exec_module(module)
                # Environment changes after import must not alter capture policy.
                os.environ["VLLM_MACH_MXFP8_BF16_GDN_PDL"] = "1" if setting == "0" else "0"
                os.environ["VLLM_MACH_MXFP8_BF16_GDN_GROUPED"] = "1" if grouped == "0" else "0"
                self.assertEqual(module.supports_shape(128, 32), grouped == "1")
                for override in (None, False, True):
                    expected = setting == "1" if override is None else override
                    args, kwargs = module.decode(*([tensor] * 8), scratch,
                                                 use_pdl=override)
                    self.assertEqual(args[-1], expected)  # USE_PDL includes wait.
                    self.assertEqual(kwargs["launch_pdl"], expected)
                    self.assertEqual(args[-2], 4 if grouped == "1" else 1)
                    self.assertEqual(kwargs["num_warps"], 2 if grouped == "1" else 4)
                with self.assertRaisesRegex(TypeError, "bool or None"):
                    module.decode(*([tensor] * 8), scratch, use_pdl=1)
                with self.assertRaisesRegex(TypeError, "bool or None"):
                    module.decode(*([tensor] * 8), scratch, grouped_layout=1)

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





class PinnedResetPolicyCPU(unittest.TestCase):
    def test_pageable_body_unchanged_and_pinned_single_upload_inverse(self):
        tree = ast.parse((GDN / "worker.py").read_text())
        original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_allocated")
        self.assertEqual(hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
                         "db27f9e7f6d32b07811428e406b513d584439ae2bd15853fbb11acf43e371fd8")
        import copy
        pinned = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_allocated_pinned"))
        pair = ast.parse('host_ids = torch.tensor(unique,dtype=torch.int32,device="cpu",pin_memory=True)\nids_by_device[key[3]] = host_ids.to(dev,non_blocking=True)').body
        changes = 0
        for owner in ast.walk(pinned):
            for field, value in ast.iter_fields(owner):
                if isinstance(value, list):
                    for i in range(len(value)-1):
                        if ast.dump(ast.Module(body=value[i:i+2], type_ignores=[]), include_attributes=False) == ast.dump(ast.Module(body=pair, type_ignores=[]), include_attributes=False):
                            value[i:i+2] = ast.parse('ids_by_device[key[3]] = torch.tensor(unique,dtype=torch.int32,device=dev)').body
                            changes += 1
                            break
        pinned.name = original.name
        self.assertEqual(changes, 1)
        self.assertEqual(ast.dump(pinned, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_startup_selector_only_qualified_bf16_profile_and_foreign_owner(self):
        worker = load_worker()
        ns = types.SimpleNamespace
        target = ns(_mach_mxfp8_contract=dict(mode="production", ssm_dtype="bfloat16", ordered_gdn=True,
                                             kv_dtype="fp8_e4m3", compact_greedy_head=True),
                    vllm_config=ns(cache_config=ns(mamba_ssm_cache_dtype="bfloat16")))
        env = {"VLLM_MACH_PROFILE": "qwen35-4b-mxfp8-champion-v1", "VLLM_HYBRID_NVFP4_LM_HEAD_CUDAGRAPH": "0"}
        with patch.dict(os.environ, env, clear=True):
            original = worker._reset_allocated
            self.assertIs(original, worker._PAGEABLE_RESET)
            worker._install_reset_policy(target)
            self.assertIs(worker._reset_allocated, worker._reset_allocated_pinned)
            for key, value in (("mode", "quality"), ("ssm_dtype", "float32"), ("ordered_gdn", False),
                               ("kv_dtype", "bfloat16"), ("compact_greedy_head", False)):
                old = target._mach_mxfp8_contract[key]; target._mach_mxfp8_contract[key] = value
                worker._install_reset_policy(target)
                self.assertIs(worker._reset_allocated, original)
                target._mach_mxfp8_contract[key] = old
            target.vllm_config.cache_config.mamba_ssm_cache_dtype = "float32"
            worker._install_reset_policy(target); self.assertIs(worker._reset_allocated, original)
            target.vllm_config.cache_config.mamba_ssm_cache_dtype = "bfloat16"
            with patch.dict(os.environ, {"VLLM_HYBRID_NVFP4_LM_HEAD_CUDAGRAPH": "1"}):
                worker._install_reset_policy(target); self.assertIs(worker._reset_allocated, original)
            for profile in ("qwen35-2b-mxfp8-champion-v1", "block-fp8", "other"):
                with patch.dict(os.environ, {"VLLM_MACH_PROFILE": profile}):
                    worker._install_reset_policy(target); self.assertIs(worker._reset_allocated, original)
            # Even a callable claiming equality cannot be overwritten by the selector.
            class Foreign:
                def __call__(self, *args): pass
                def __eq__(self, other): return True
            foreign = Foreign(); worker._reset_allocated = foreign
            with self.assertRaisesRegex(RuntimeError, "another hook"):
                worker._install_reset_policy(target)
            self.assertIs(worker._reset_allocated, foreign)

    def test_actual_reset_bodies_fresh_pin_order_counters_and_original_errors(self):
        worker = load_worker()
        package = "_mach_reset_policy_cpu_fixture"
        parent = types.ModuleType(package); parent.__path__ = []
        kernels = types.ModuleType(package + ".metadata_reset"); parent.metadata_reset = kernels
        torch = types.ModuleType("torch"); torch.int32 = object(); torch.device = lambda value: value
        events = []; hosts = []
        class Tensor:
            def __init__(self, values, device, pinned): self.values, self.device, self.pinned = list(values), device, pinned
            def to(self, device, non_blocking=False):
                events.append(("to", id(self), device, non_blocking))
                assert self.device == "cpu" and self.pinned and non_blocking
                return Tensor(self.values, device, False)
        def tensor(values, **kwargs):
            self.assertIs(kwargs["dtype"], torch.int32)
            value = Tensor(values, kwargs["device"], kwargs.get("pin_memory", False))
            events.append(("tensor", list(values), value.device, value.pinned))
            if value.pinned: hosts.append(value)
            return value
        torch.tensor = tensor
        def reset(ids, scratch, pages): events.append(("kernel", scratch, list(ids.values), ids.device, pages))
        kernels.reset_metadata = reset
        worker.__package__ = package
        worker.__spec__ = importlib.util.spec_from_file_location(package + ".worker", GDN / "worker.py")
        with patch.dict(sys.modules, {"torch": torch, package: parent, package + ".metadata_reset": kernels}):
            for ids, ready in (([9, 1, 9, 3], True), ([1, 15], True), ([], True), ([3], False), ([16], True)):
                outcomes = []
                for fn in (worker._PAGEABLE_RESET, worker._reset_allocated_pinned):
                    worker._READY = ready
                    worker._VIEW_SCRATCH = {(i, (16, 32, 4), (), "cuda:0", "fixture"): i for i in range(8)}
                    worker._ALLOCATION_RESETS = {"new": 0, "cached": 0, "ids": 0}
                    events.clear(); error = None
                    try: fn(ids, "new")
                    except RuntimeError as exc: error = (type(exc).__name__, str(exc))
                    outcomes.append(([e for e in events if e[0] == "kernel"], dict(worker._ALLOCATION_RESETS), error))
                    if ids and ready and max(ids) < 16:
                        self.assertEqual(len(outcomes[-1][0]), 8)
                        self.assertEqual([e[1] for e in outcomes[-1][0]], list(range(8)))
                        if fn is worker._reset_allocated_pinned:
                            self.assertEqual([e[0] for e in events], ["tensor", "to"] + ["kernel"]*8)
                self.assertEqual(*outcomes)
            worker._READY = True; worker._VIEW_SCRATCH = {(i, (16,), (), "cuda:0", "fixture"): i for i in range(8)}
            start = len(hosts)
            worker._reset_allocated_pinned([1, 3], "cached"); worker._reset_allocated_pinned([1, 3], "cached")
            self.assertEqual(len(hosts)-start, 2)
            self.assertIsNot(hosts[-1], hosts[-2])

    def test_policy_installed_before_original_runner_lifecycle(self):
        tree = ast.parse((GDN / "worker.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install_runtime")
        statements = [ast.unparse(n) for n in node.body]
        index = lambda text: next(i for i, value in enumerate(statements) if text in value)
        self.assertLess(index("_check_config(worker)"), index("_install_reset_policy(worker)"))
        self.assertLess(index("_RUN_MODE, _QUALITY_ROW"), index("_install_reset_policy(worker)"))
        self.assertLess(index("_install_reset_policy(worker)"), index("_install_runner_lifecycle()"))


if __name__ == "__main__":
    unittest.main()
