"""CPU contracts for rebuilding source-bound Triton launch configurations."""
from __future__ import annotations

import ast
from functools import lru_cache
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE_FILE = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/compile_choices.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mach_compile_choices_cpu", SOURCE_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def generated_source(name="triton_poi_fused_fixture_0"):
    return f'''
@triton_heuristics.pointwise(
    size_hints={{'x': 65536}},
    filename=__file__,
    triton_meta={{'signature': {{'in_ptr0': '*bf16', 'out_ptr0': '*bf16',
                                'xnumel': 'i32', 'XBLOCK': 'constexpr'}},
                 'device': DeviceProperties(type='cuda', index=0, cc=120),
                 'constants': {{}}, 'enable_fp_fusion': True,
                 'launch_pdl': False, 'disable_ftz': False,
                 'configs': [{{(0,): [['tt.divisibility', 16]]}}]}},
    inductor_meta={{'kernel_name': '{name}', 'dynamic_scale_rblock': True}})
@triton.jit
def {name}(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    value = tl.load(in_ptr0 + xindex, xindex < xnumel)
    squared = value * value
    result = tl.sum(squared, axis=0)
    tl.store(out_ptr0, result)
'''


def function_source(source):
    fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    return ast.get_source_segment(source, fn)


class CompileChoiceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def fixture(self):
        module = self.module
        bindings = [module.recipe_from_source(
            generated_source(f"triton_poi_fused_fixture_{i}"),
            f"triton_poi_fused_fixture_{i}",
            {"XBLOCK": 512, "num_warps": 4, "num_stages": 1,
             "found_by_coordesc": False}, f"choice_{i:02d}") for i in range(23)]
        module._POLICY = {"format_version": 1, "torch_version_prefix": "2.13.",
                          "bindings": bindings, "canonical_conflicts": [],
                          "autotuner_init_sha256": "fixture"}
        meta = {"signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16",
                              "xnumel": "i32", "XBLOCK": "constexpr"},
                "constants": {}, "configs": [{(0,): [["tt.divisibility", 16]]}],
                "device": types.SimpleNamespace(type="cuda", cc=120),
                "enable_fp_fusion": True, "launch_pdl": False, "disable_ftz": False}
        return bindings, meta

    def test_public_policy_has_23_bindings_12_keys_and_explicit_conflicts(self):
        policy = self.module.load_policy()
        self.assertEqual(len(policy["bindings"]), 23)
        self.assertEqual(len({row["semantic_key"] for row in policy["bindings"]}), 12)
        self.assertEqual(len(policy["canonical_conflicts"]), 3)
        conflict_sets = {tuple(row["bindings"]) for row in policy["canonical_conflicts"]}
        self.assertEqual(conflict_sets, {
            ("choice_02", "choice_08", "choice_14", "choice_16"),
            ("choice_05", "choice_15"), ("choice_03", "choice_17")})
        text = self.module._POLICY_FILE.read_text()
        for forbidden in ("/task", "/legacy", ".best_config", "cache_key", "source_relative_path"):
            self.assertNotIn(forbidden, text)
        for row in policy["bindings"]:
            self.assertEqual(row["semantic_key"], self.module._json_hash(row["boundary"]))
            self.assertEqual(row["boundary"]["architecture"], {"type": "cuda", "cc": 120})

    def test_function_key_ignores_name_decorators_comments_but_keeps_reduction(self):
        module = self.module
        source = generated_source()
        original = module.canonical_function_sha256(source)
        changed = source.replace("triton_poi_fused_fixture_0", "arbitrary_new_name")
        changed = changed.replace("filename=__file__", "filename='/arbitrary/new/location'")
        changed = changed.replace("    value =", "    # irrelevant compiler path comment\n    value =")
        self.assertEqual(original, module.canonical_function_sha256(changed))
        changed_math = source.replace("tl.sum(squared, axis=0)", "tl.max(squared, axis=0)")
        self.assertNotEqual(original, module.canonical_function_sha256(changed_math))
        changed_args = source.replace("XBLOCK: tl.constexpr", "XBLOCK")
        self.assertNotEqual(original, module.canonical_function_sha256(changed_args))

    def test_metadata_fingerprint_binds_fp_flags_constants_and_specialization(self):
        _, meta = self.fixture()
        source = function_source(generated_source())
        original = self.module.descriptor(source, {"x": 65536}, meta)
        for field, value in (("enable_fp_fusion", False), ("disable_ftz", True),
                             ("constants", {"xnumel": 65536}),
                             ("configs", [{(0,): [["tt.divisibility", 8]]}])):
            changed = self.module.descriptor(source, {"x": 65536}, {**meta, field: value})
            self.assertNotEqual(original, changed)
        changed_device = self.module.descriptor(source, {"x": 65536}, {
            **meta, "device": types.SimpleNamespace(type="cuda", cc=90)})
        self.assertNotEqual(original, changed_device)
        self.assertNotEqual(original, self.module.descriptor(source, {"x": 131072}, meta))

    def test_conflicting_canonical_choices_need_exact_site(self):
        bindings, meta = self.fixture()
        bindings[1]["selected_fields"] = {"XBLOCK": 256, "num_warps": 4,
                                          "num_stages": 1, "found_by_coordesc": False}
        source = function_source(generated_source("renamed"))
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch, "exact site"):
            self.module.select_recipe(source, {"x": 65536}, meta, kernel_alias="renamed")
        for i in (0, 1):
            row = self.module.select_recipe(source, {"x": 65536}, meta,
                                           kernel_alias=bindings[i]["kernel_alias"],
                                           kernel_site=bindings[i]["kernel_site"])
            self.assertIs(row, bindings[i])

    def test_known_geometry_math_or_fp_drift_raises_unknown_kernels_pass(self):
        bindings, meta = self.fixture()
        source = function_source(generated_source())
        for changed_source, changed_meta in (
                (source.replace("tl.sum", "tl.max"), meta),
                (source, {**meta, "enable_fp_fusion": False})):
            with self.assertRaisesRegex(self.module.CompileChoiceMismatch, "boundary changed"):
                self.module.select_recipe(changed_source, {"x": 65536}, changed_meta,
                                          kernel_alias=bindings[0]["kernel_alias"])
        # A different geometry is outside the fixed recipe and keeps native
        # heuristics; an unrelated kernel is not a fatal global allowlist miss.
        self.assertIsNone(self.module.select_recipe(source, {"x": 131072}, meta,
                                                  kernel_alias=bindings[0]["kernel_alias"]))
        self.assertIsNone(self.module.select_recipe(source.replace("tl.sum", "tl.max"),
                                                  {"x": 65536}, meta, kernel_alias="new_kernel"))
        self.assertEqual(self.module.inspect()["unknown_constructions"], 2)

    def runtime(self):
        _, meta = self.fixture()
        module = self.module
        class Config:
            def __init__(self, kwargs, *, num_warps, num_stages):
                self.kwargs = kwargs
                self.num_warps = num_warps
                self.num_stages = num_stages
        class Autotuner:
            def __init__(self, fn, triton_meta, configs, save_cache_hook=None,
                         heuristic_type=None, size_hints=None, inductor_meta=None):
                self.fn = fn
                self.triton_meta = triton_meta
                self.inductor_meta = inductor_meta
                # Simulate Torch's weaker lookup overriding the incoming list.
                self.configs = [Config({"XBLOCK": 128}, num_warps=8, num_stages=1)]
                self.received_configs = configs
                self.compile_results = []
        class AsyncCompile:
            @staticmethod
            @lru_cache(1)
            def process_pool():
                return object()
        torch = types.ModuleType("torch")
        torch.__version__ = "2.13.0+test"
        torch.__path__ = []
        inductor = types.ModuleType("torch._inductor")
        inductor.__path__ = []
        config = types.ModuleType("torch._inductor.config")
        config.compile_threads = 8
        async_compile = types.ModuleType("torch._inductor.async_compile")
        async_compile.AsyncCompile = AsyncCompile
        async_compile._pool_set = set()
        runtime = types.ModuleType("torch._inductor.runtime")
        runtime.__path__ = []
        heuristics = types.ModuleType("torch._inductor.runtime.triton_heuristics")
        heuristics.CachingAutotuner = Autotuner
        triton = types.ModuleType("triton")
        triton.Config = Config
        modules = {entry.__name__: entry for entry in (
            torch, inductor, config, async_compile, runtime, heuristics, triton)}
        module._POLICY["autotuner_init_sha256"] = module.canonical_function_sha256(
            inspect.getsource(Autotuner.__init__))
        return modules, Autotuner, Config, config, AsyncCompile, meta

    def test_install_pins_only_known_ops_and_validates_all_23_bindings(self):
        modules, Autotuner, Config, config, _, meta = self.runtime()
        module = self.module
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
            self.assertTrue(module.install())
            self.assertFalse(module.install())
            self.assertEqual(config.compile_threads, 1)
            self.assertEqual(os.environ["TORCHINDUCTOR_COMPILE_THREADS"], "1")
            tuners = []
            for row in module._POLICY["bindings"]:
                fn = types.SimpleNamespace(__name__=row["kernel_alias"],
                    src=function_source(generated_source(row["kernel_alias"])))
                tuners.append(Autotuner(fn, meta, [Config({"XBLOCK": 64}, num_warps=4, num_stages=1)],
                              size_hints={"x": 65536},
                              inductor_meta={"kernel_name": row["kernel_site"],
                                             "dynamic_scale_rblock": True}))
            self.assertEqual(len(module.verify_coverage()["seen_bindings"]), 23)
            for tuner in tuners:
                self.assertEqual(tuner.configs[0].kwargs, {"XBLOCK": 512})
                self.assertFalse(tuner.inductor_meta["dynamic_scale_rblock"])
                self.assertFalse(tuner.inductor_meta["coordinate_descent_tuning"])
            unknown = Autotuner(types.SimpleNamespace(__name__="new_kernel", src="def new_kernel(x):\n    return x"),
                                meta, [], size_hints={"x": 1}, inductor_meta={"dynamic_scale_rblock": True})
            self.assertTrue(unknown.inductor_meta["dynamic_scale_rblock"])
            self.assertEqual(unknown.configs[0].kwargs, {"XBLOCK": 128})
            self.assertEqual(module.verify_coverage()["unknown_constructions"], 1)
            # Detect a later launch mutation even if every constructor matched.
            tuners[0].configs[0].kwargs["XBLOCK"] = 256
            with self.assertRaisesRegex(module.CompileChoiceMismatch, "configuration changed"):
                module.verify_coverage()

    def test_compiled_result_launch_is_checked_after_configs_are_consumed(self):
        modules, Autotuner, Config, _, _, meta = self.runtime()
        module = self.module
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
            module.install()
            tuners = []
            for row in module._POLICY["bindings"]:
                tuner = Autotuner(types.SimpleNamespace(__name__=row["kernel_alias"],
                                  src=function_source(generated_source(row["kernel_alias"]))),
                                  meta, [], size_hints={"x": 65536},
                                  inductor_meta={"kernel_name": row["kernel_site"]})
                tuner.compile_results = [types.SimpleNamespace(config=tuner.configs[0])]
                tuner.configs = None
                tuners.append(tuner)
            module.verify_coverage()
            tuners[0].compile_results.append(types.SimpleNamespace(
                config=Config({"XBLOCK": 512}, num_warps=8, num_stages=1)))
            with self.assertRaisesRegex(module.CompileChoiceMismatch, "configuration changed"):
                module.verify_coverage()

    def test_install_rejects_active_process_pool_and_constructor_drift(self):
        modules, _, _, _, AsyncCompile, _ = self.runtime()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
            AsyncCompile.process_pool()
            with self.assertRaisesRegex(RuntimeError, "process pool exists"):
                self.module.install()
            AsyncCompile.process_pool.cache_clear()
            self.module._POLICY["autotuner_init_sha256"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "constructor differs"):
                self.module.install()

    def test_empty_or_partial_coverage_cannot_be_reported_complete(self):
        self.fixture()
        self.module._INSTALLED = True
        self.module._BINDING_HITS["choice_00"] = 1
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch, "coverage incomplete"):
            self.module.verify_coverage()

    def test_explicit_partial_coverage_still_checks_observed_launch(self):
        modules, Autotuner, _, _, _, meta = self.runtime()
        module = self.module
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
            module.install()
            row = module._POLICY["bindings"][0]
            tuner = Autotuner(types.SimpleNamespace(__name__=row["kernel_alias"],
                src=function_source(generated_source(row["kernel_alias"]))), meta, [],
                size_hints={"x": 65536}, inductor_meta={"kernel_name": row["kernel_site"]})
            with self.assertRaisesRegex(module.CompileChoiceMismatch, "coverage incomplete"):
                module.verify_coverage()
            receipt = module.verify_coverage(require_complete=False)
            self.assertEqual(receipt["seen_bindings"], [row["id"]])
            self.assertEqual(len(receipt["missing_bindings"]), 22)
            tuner.configs[0].kwargs["XBLOCK"] = 256
            with self.assertRaisesRegex(module.CompileChoiceMismatch, "configuration changed"):
                module.verify_coverage(require_complete=False)

    def test_quality_policies_are_independent_and_frozen_at_loading(self):
        production = self.module.load_policy()
        production_hash = self.module._json_hash(production)
        for rows, count in ((32, 23), (64, 22)):
            module = load_module()
            with patch.dict(os.environ, {"VLLM_MACH_MXFP8_MODE": "quality",
                                         "VLLM_MACH_MXFP8_QUALITY_ROWS": str(rows)}):
                policy = module.load_policy()
                self.assertEqual(len(policy["bindings"]), count)
                self.assertEqual(policy["profile_mode"], f"quality{rows}")
                self.assertEqual(module.inspect()["policy_mode"], f"quality{rows}")
                self.assertEqual({row["boundary"]["architecture"]["cc"]
                                  for row in policy["bindings"]}, {120})
                self.assertTrue(all(row.get("accepted_source_sha256")
                                    and row.get("accepted_choice_sha256")
                                    for row in policy["bindings"]))
                for row in policy["bindings"]:
                    self.assertEqual(row["semantic_key"], module._json_hash(row["boundary"]))
                module._INSTALLED = True
                with self.assertRaisesRegex(module.CompileChoiceMismatch, "coverage incomplete"):
                    module.verify_coverage()
                with patch.dict(os.environ, {"VLLM_MACH_MXFP8_QUALITY_ROWS": str(96 - rows)}):
                    with self.assertRaisesRegex(RuntimeError, "mode changed"):
                        module.load_policy()
        self.assertEqual(self.module._json_hash(self.module.load_policy()), production_hash)


class RealTritonMetadataTests(unittest.TestCase):
    """Run with the locked Torch/Triton wheel; never select or allocate a GPU."""
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import triton
            import triton.language as tl
        except ImportError:
            raise unittest.SkipTest("Real Torch 2.13/Triton 3.7.1 runtime required")
        if not torch.__version__.split("+", 1)[0].startswith("2.13.") or triton.__version__ != "3.7.1":
            raise unittest.SkipTest("Test binds the locked Torch 2.13/Triton 3.7.1 ABI")
        cls.torch, cls.triton, cls.tl = torch, triton, tl

    def setUp(self):
        self.module = load_module()

    def test_all_mode_signature_type_roundtrips_keep_exact_policy_key(self):
        module, tl = self.module, self.tl
        paths = [module._POLICY_FILE, module._POLICY_FILE.with_name("compile_choices_quality32.json"),
                 module._POLICY_FILE.with_name("compile_choices_quality64.json")]
        for path in paths:
            for row in json.loads(path.read_text())["bindings"]:
                original = row["boundary"]
                converted = {name: tl.str_to_ty(value, None) for name, value in original["signature"].items()}
                normalized = module._signature_encoded(converted)
                self.assertEqual(normalized, original["signature"], row["id"])
                self.assertEqual(module._json_hash({**original, "signature": normalized}), row["semantic_key"])

    def test_real_types_preserve_pointer_qualifiers_and_typed_constants(self):
        module, tl = self.module, self.tl
        scalars = ("bf16", "fp16", "fp32", "fp64", "i32", "i64", "u8")
        for symbol in scalars:
            value = tl.str_to_ty(symbol, None)
            for token in (symbol, "*" + symbol, "*k" + symbol):
                actual = tl.str_to_ty(token, None)
                self.assertEqual(module._signature_encoded({"arg": actual}), {"arg": token})
            self.assertNotEqual(module._encoded(value), module._encoded(symbol))
        with self.assertRaisesRegex(TypeError, "address space"):
            module._signature_encoded({"arg": tl.pointer_type(tl.bfloat16, address_space=3)})

    def test_known_source_matches_real_signature_and_still_rejects_drift(self):
        module, tl = self.module, self.tl
        row = module.recipe_from_source(generated_source(), "triton_poi_fused_fixture_0",
            {"XBLOCK": 512, "num_warps": 4, "num_stages": 1, "found_by_coordesc": False}, "choice_00")
        module._POLICY = {"bindings": [row]}
        meta = {"signature": {key: tl.str_to_ty(value, None) for key, value in row["boundary"]["signature"].items()},
                "constants": {}, "configs": [{(0,): [["tt.divisibility", 16]]}],
                "device": types.SimpleNamespace(type="cuda", cc=120),
                "enable_fp_fusion": True, "launch_pdl": False, "disable_ftz": False}
        source = function_source(generated_source())
        self.assertIs(module.select_recipe(source, {"x": 65536}, meta, kernel_alias=row["kernel_alias"]), row)
        for changed in ({**meta, "signature": {**meta["signature"], "in_ptr0": tl.pointer_type(tl.bfloat16, const=True)}},
                        {**meta, "constants": {"dtype": tl.float32}}, {**meta, "enable_fp_fusion": False}):
            with self.assertRaisesRegex(module.CompileChoiceMismatch, "boundary changed"):
                module.select_recipe(source, {"x": 65536}, changed, kernel_alias=row["kernel_alias"])
        self.assertIsNone(module.select_recipe("def unrelated(x):\n    return x", {"x": 65536},
            {**meta, "constants": {"dtype": tl.float32, "outside_schema": object()}}, kernel_alias="unrelated"))

    def test_real_torch_custom_kernel_metadata_path_keeps_dtype_constant_native(self):
        # The real Torch cached_autotune passes constants through unchanged.
        # A recorder replaces only the GPU-owning constructor; JIT parsing,
        # config/cache handling and the metadata handoff are the actual APIs.
        from torch._inductor.runtime import triton_heuristics as h
        module, triton, tl = self.module, self.triton, self.tl
        def custom_gdn_dtype_fixture(in_ptr0, out_ptr0, n, DTYPE):
            pass
        fn = triton.jit(custom_gdn_dtype_fixture)
        meta = {"signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16", "n": "i32", "DTYPE": "constexpr"},
                "constants": {"DTYPE": tl.float32}, "configs": [],
                "device": types.SimpleNamespace(type="cuda", cc=120)}
        config = triton.Config({}, num_warps=4, num_stages=1)
        seen = {}
        class Recorder:
            def __init__(self, kernel, **kwargs):
                seen.update(kwargs)
                self.row = module.select_recipe(kernel.src, kwargs["size_hints"], kwargs["triton_meta"],
                                                kernel_alias=kernel.__name__)
        before = self.torch.cuda.is_initialized()
        result = h.cached_autotune(None, [config], triton_meta=meta,
            heuristic_type=h.HeuristicType.POINTWISE, inductor_meta={"force_disable_caches": True},
            custom_kernel=True, caching_autotuner_cls=Recorder)(fn)
        self.assertIsNone(result.row)
        self.assertIs(seen["triton_meta"]["constants"]["DTYPE"], tl.float32)
        self.assertIs(seen["configs"][0], config)
        self.assertTrue(seen["custom_kernel"])
        self.assertEqual(self.torch.cuda.is_initialized(), before)


if __name__ == "__main__":
    unittest.main()
