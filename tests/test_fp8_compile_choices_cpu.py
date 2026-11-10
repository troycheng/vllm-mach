"""CPU contracts for independent stock RMS launch recipes; no Torch install."""
import ast
from functools import lru_cache
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCE = ROOT / "src/vllm_mach/fp8/compile_choices.py"


def load_module():
    spec = importlib.util.spec_from_file_location("vllm_mach.fp8._choices_cpu", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generated_source(name="rms_fixture_0"):
    return f'''
@triton_heuristics.reduction(
    size_hints={{'x': 2048, 'r0_': 4096}},
    triton_meta={{'signature': {{'in_ptr0': '*bf16', 'out_ptr0': '*bf16'}},
                 'device': DeviceProperties(type='cuda', cc=120),
                 'constants': {{}}, 'configs': []}},
    inductor_meta={{'kernel_name': '{name}'}})
@triton.jit
def {name}(in_ptr0, out_ptr0, R0_BLOCK: tl.constexpr):
    value = tl.load(in_ptr0 + tl.arange(0, R0_BLOCK))
    reduced = tl.sum(value * value, axis=0)
    tl.store(out_ptr0, reduced)
'''


class CompileChoicesTests(unittest.TestCase):
    def setUp(self):
        self.m = load_module()
        self.env = patch.dict(os.environ, {"VLLM_MACH_FP8_COMPILE_MODE": "quality4"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def fixture(self):
        source = generated_source()
        fields = {"XBLOCK": 1, "R0_BLOCK": 4096, "num_warps": 16,
                  "num_stages": 1, "found_by_coordesc": False}
        r = self.m.recipe_from_source(source, "rms_fixture_0", fields, "fixture")
        row = {k: r[k] for k in ("id", "selected_fields")}
        row["kernel_aliases"] = [r["kernel_alias"]]
        row["kernel_sites"] = [r["kernel_site"]]
        row["variants"] = [{"name": "stock", "boundary": r["boundary"],
                             "semantic_key": r["semantic_key"]}]
        self.m._MODE = "quality4"
        self.m._POLICY = {"format_version": 2, "binding_semantics": "exact_boundary_and_selected_config", "profile_mode": "quality4",
                          "torch_version": "2.14.0+cu130", "triton_version": "3.8.0",
                          "autotuner_init_sha256": "fixture",
                          "bindings": [row]}
        meta = {"signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16"},
                "device": types.SimpleNamespace(type="cuda", cc=120),
                "constants": {}, "configs": []}
        return source, meta, row

    def select(self, source, meta, *, shape=None, alias="rms_fixture_0", site="rms_fixture_0"):
        return self.m.select_recipe(source, shape or {"x": 2048, "r0_": 4096}, meta,
                                    kernel_alias=alias, kernel_site=site)

    def test_public_modes_are_separate_stock_choices(self):
        snapshots = {}
        for mode in ("quality4", "quality32", "quality64", "production"):
            m = load_module()
            with patch.dict(os.environ, {"VLLM_MACH_FP8_COMPILE_MODE": mode}):
                p = m.load_policy()
                snapshots[mode] = p
                self.assertEqual(len(p["bindings"]), 3)
                self.assertEqual(p["binding_semantics"], "exact_boundary_and_selected_config")
                self.assertEqual(sum(len(r["provenance_bindings"]) for r in p["bindings"]), 6)
                keys = [v["semantic_key"] for r in p["bindings"] for v in r["variants"]]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(p["torch_version"], "2.14.0+cu130")
                self.assertEqual(p["triton_version"], "3.8.0")
                for r in p["bindings"]:
                    for v in r["variants"]:
                        self.assertEqual(v["semantic_key"], m._json_hash(v["boundary"]))
                        self.assertEqual(v["boundary"]["architecture"], {"type": "cuda", "cc": 120})
                text = json.dumps(p)
                for forbidden in (".best_config", "/task", "cache_key", "source_relative_path"):
                    self.assertNotIn(forbidden, text)
        self.assertNotEqual(snapshots["quality4"]["bindings"], snapshots["quality32"]["bindings"])
        q4 = snapshots["quality4"]["bindings"]
        terminal = next(r for r in q4 if len(r["variants"]) == 2)
        self.assertEqual(terminal["selected_fields"]["R0_BLOCK"], 1024)
        self.assertEqual(terminal["selected_fields"]["num_warps"], 8)

    def test_quality32_actual_final_norm_matches_stock_without_phantom_coverage(self):
        fixtures = json.loads((ROOT / "tests/data/inductor033-block-quality32-norms.json").read_text())
        with patch.dict(os.environ, {"VLLM_MACH_FP8_COMPILE_MODE": "quality32"}):
            for role, source in fixtures.items():
                fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
                deco = next(n for n in fn.decorator_list if isinstance(n, ast.Call))
                keywords = {k.arg: k.value for k in deco.keywords}
                meta = {}
                for k, v in zip(keywords["triton_meta"].keys, keywords["triton_meta"].values):
                    key = ast.literal_eval(k)
                    meta[key] = ({x.arg: ast.literal_eval(x.value) for x in v.keywords}
                                 if key == "device" else ast.literal_eval(v))
                selected = self.m.select_recipe(ast.get_source_segment(source, fn),
                    ast.literal_eval(keywords["size_hints"]), meta,
                    kernel_alias=fn.name, kernel_site=fn.name)
                self.assertEqual((selected["selected_fields"]["R0_BLOCK"],
                                  selected["selected_fields"]["num_warps"]),
                                 (4096, 16) if role == "intermediate" else (1024, 8))
                if role == "final_inplace":
                    self.assertEqual(selected["id"], "rms_group_01")
            receipt = self.m.inspect()
            self.assertEqual(receipt["required_bindings"], 3)
            self.assertEqual(receipt["missing_bindings"], [])
            # The separately generated old out-of-place variant is accepted,
            # but its absence must not add a required runtime group.
            policy = self.m.load_policy()
            retained = next(v for row in policy["bindings"] for v in row["variants"]
                            if v["boundary"]["function_sha256"] ==
                            "0434deadf367733171da7793906321a53149af1472ecc1d7f686aeb9f195f5dc")
            row = next(row for row in policy["bindings"] if retained in row["variants"])
            self.assertEqual((row["selected_fields"]["R0_BLOCK"], row["selected_fields"]["num_warps"]), (4096, 16))

    def test_other_modes_retain_their_launch_choices(self):
        expected = {"production": [(1024, 8), (1024, 8), (4096, 16)],
                    "quality4": [(1024, 8), (1024, 8), (4096, 16)],
                    "quality64": [(4096, 16), (1024, 8), (1024, 8)]}
        for mode, choices in expected.items():
            module = load_module()
            rows = module.load_policy(mode)["bindings"]
            self.assertEqual([(r["selected_fields"]["R0_BLOCK"], r["selected_fields"]["num_warps"])
                              for r in rows], choices)

    def test_off_does_not_import_torch_or_change_env(self):
        with patch.dict(os.environ, {"VLLM_MACH_FP8_COMPILE_MODE": ""}):
            before = dict(os.environ)
            self.assertFalse(self.m.install())
            self.assertEqual(before, dict(os.environ))
            self.assertFalse(self.m.inspect()["installed"])

    def test_never_installs_mxfp8_policy(self):
        from vllm_mach.mxfp8 import compile_choices as mx8
        before = (mx8._POLICY, mx8._INSTALLED, mx8._POLICY_MODE)
        self.m.load_policy()
        self.assertEqual(before, (mx8._POLICY, mx8._INSTALLED, mx8._POLICY_MODE))

    def test_mode_change_rejected_after_load_and_in_receipt(self):
        self.m.load_policy()
        with patch.dict(os.environ, {"VLLM_MACH_FP8_COMPILE_MODE": "quality32"}):
            with self.assertRaisesRegex(RuntimeError, "mode changed"):
                self.m.load_policy()
            with self.assertRaisesRegex(RuntimeError, "mode changed"):
                self.m.inspect()

    def test_exact_source_meta_shape_required_independent_of_generated_site(self):
        source, meta, row = self.fixture()
        self.assertEqual(self.select(source, meta)["id"], row["id"])
        changes = [(source.replace("tl.sum", "tl.max"), meta, {}, "rms_fixture_0"),
                   (source, {**meta, "enable_fp_fusion": False}, {}, "rms_fixture_0"),
                   (source, {**meta, "constants": {"r0_numel": 2560}}, {}, "rms_fixture_0"),
                   (source, {**meta, "device": types.SimpleNamespace(type="cuda", cc=90)}, {}, "rms_fixture_0"),
                   (source, meta, {"x": 8192, "r0_": 4096}, "rms_fixture_0")]
        for src, md, shape, site in changes:
            with self.assertRaises(self.m.CompileChoiceMismatch):
                self.select(src, md, shape=shape, site=site)

    def test_new_alias_hits_only_actual_semantic_group(self):
        source, meta, row = self.fixture()
        name = "triton_red_fused__to_copy_add_fused_add_rms_norm_per_token_group_fp8_quant_permute_2"
        renamed = source.replace("rms_fixture_0", name)
        selected = self.select(renamed, meta, alias=name, site=name)
        self.assertEqual(selected["id"], row["id"])
        self.assertEqual(selected["selected_fields"], row["selected_fields"])
        self.assertEqual(selected["observed_site"], name)
        self.assertEqual(self.m.inspect()["binding_hits"], {row["id"]: 1})
        self.assertEqual(self.m.inspect()["missing_bindings"], [])
        for src, md in [(renamed.replace("tl.sum", "tl.max"), meta),
                        (renamed, {**meta, "enable_fp_fusion": False})]:
            with self.assertRaises(self.m.CompileChoiceMismatch):
                self.select(src, md, alias=name, site=name)

    def test_conflicting_semantic_configs_are_rejected(self):
        source, meta, row = self.fixture()
        duplicate = {**row, "id": "conflicting", "selected_fields": {
            **row["selected_fields"], "num_warps": 4}}
        self.m._POLICY["bindings"].append(duplicate)
        with self.assertRaises(self.m.CompileChoiceMismatch):
            self.select(source, meta)
        self.assertFalse(self.m.inspect()["binding_hits"])

    def test_loader_rejects_reused_semantic_with_conflicting_config(self):
        policy = json.loads((self.m._DATA / "compile_choices_quality4.json").read_text())
        duplicate = {**policy["bindings"][0], "id": "conflicting",
                     "selected_fields": {**policy["bindings"][0]["selected_fields"], "num_warps": 4}}
        policy["bindings"].append(duplicate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "compile_choices_quality4.json").write_text(json.dumps(policy))
            self.m._DATA = path
            with self.assertRaisesRegex(RuntimeError, "conflicting.*semantic"):
                self.m.load_policy()

    def test_complete_coverage_requires_each_real_group(self):
        source, meta, row = self.fixture()
        sources = [source, source.replace("value * value", "value + value"),
                   source.replace("tl.sum", "tl.max")]
        for index, src in enumerate(sources[1:], 1):
            recipe = self.m.recipe_from_source(src, "rms_fixture_0", row["selected_fields"], str(index))
            self.m._POLICY["bindings"].append({
                **row, "id": str(index), "variants": [{"name": "stock",
                "boundary": recipe["boundary"], "semantic_key": recipe["semantic_key"]}]})
        self.m._INSTALLED = True
        self.select(source, meta)
        self.assertEqual(self.m.inspect()["missing_bindings"], ["1", "2"])
        with self.assertRaisesRegex(self.m.CompileChoiceMismatch, "coverage incomplete"):
            self.m.verify_coverage()
        for src in sources[1:]:
            self.select(src, meta)
        self.assertEqual(self.m.inspect()["missing_bindings"], [])
        self.assertEqual(self.m.inspect()["binding_hits"], {row["id"]: 1, "1": 1, "2": 1})

    def test_unrelated_kernel_is_untouched(self):
        self.fixture()
        self.assertIsNone(self.select("def unrelated(x):\n    return x", {}, alias="unrelated", site="unrelated"))
        self.assertEqual(self.m.inspect()["unknown_constructions"], 1)

    def runtime(self):
        source, meta, row = self.fixture()
        class Config:
            def __init__(self, kwargs, *, num_warps, num_stages):
                self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages
        class Autotuner:
            def __init__(self, fn, triton_meta, configs, size_hints=None, inductor_meta=None):
                self.fn = fn
                self.received = configs
                self.inductor_meta = inductor_meta
                self.configs = [Config({"XBLOCK": 128}, num_warps=4, num_stages=1)]
                self.compile_results = []
        class AsyncCompile:
            @staticmethod
            @lru_cache(1)
            def process_pool():
                return object()
        modules = {}
        for name in ("torch", "torch._inductor", "torch._inductor.config",
                     "torch._inductor.async_compile", "torch._inductor.runtime",
                     "torch._inductor.runtime.triton_heuristics", "triton"):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules["torch"].__version__ = "2.14.0+cu130"
        modules["torch._inductor.config"].compile_threads = 8
        modules["torch._inductor.async_compile"].AsyncCompile = AsyncCompile
        modules["torch._inductor.async_compile"]._pool_set = set()
        modules["torch._inductor.runtime.triton_heuristics"].CachingAutotuner = Autotuner
        modules["triton"].Config = Config
        modules["triton"].__version__ = "3.8.0"
        self.m._POLICY["autotuner_init_sha256"] = self.m.canonical_function_sha256(inspect.getsource(Autotuner.__init__))
        return modules, Autotuner, Config, AsyncCompile, source, meta, row

    def tuner(self, cls, source, meta):
        return cls(types.SimpleNamespace(__name__="rms_fixture_0", src=source), meta, [],
                   size_hints={"x": 2048, "r0_": 4096},
                   inductor_meta={"kernel_name": "rms_fixture_0", "dynamic_scale_rblock": True})

    def test_install_config_receipt_and_compiled_mutation(self):
        modules, cls, config, _, source, meta, row = self.runtime()
        with patch.dict(sys.modules, modules):
            self.assertTrue(self.m.install())
            self.assertFalse(self.m.install())
            self.assertEqual(modules["torch._inductor.config"].compile_threads, 1)
            tuner = self.tuner(cls, source, meta)
            self.assertFalse(tuner.inductor_meta["dynamic_scale_rblock"])
            receipt = self.m.verify_coverage()
            self.assertEqual(receipt["actual_choices"][0]["actual_configs"], [row["selected_fields"]])
            self.assertEqual(os.environ["TORCHINDUCTOR_COMPILE_THREADS"], "1")
            tuner.compile_results = [types.SimpleNamespace(config=tuner.configs[0])]
            tuner.configs = None
            self.assertEqual(self.m.verify_coverage()["actual_choices"][0]["state"], "compiled")
            tuner.compile_results[0].config.kwargs["R0_BLOCK"] = 1024
            with self.assertRaisesRegex(self.m.CompileChoiceMismatch, "actual launch config changed"):
                self.m.verify_coverage()

    def test_unknown_autotuner_retains_native_config_and_controls(self):
        modules, cls, _, _, _, meta, _ = self.runtime()
        with patch.dict(sys.modules, modules):
            self.m.install()
            tuner = cls(types.SimpleNamespace(__name__="other", src="def other(x):\n    return x"),
                        meta, [], size_hints={"x": 1}, inductor_meta={"dynamic_scale_rblock": True})
            self.assertEqual(tuner.configs[0].kwargs, {"XBLOCK": 128})
            self.assertTrue(tuner.inductor_meta["dynamic_scale_rblock"])
            with self.assertRaisesRegex(self.m.CompileChoiceMismatch, "coverage incomplete"):
                self.m.verify_coverage()
            self.m.verify_coverage(require_complete=False)

    def test_pool_or_torch_or_constructor_drift_rejected(self):
        modules, _, _, pool, _, _, _ = self.runtime()
        with patch.dict(sys.modules, modules):
            pool.process_pool()
            with self.assertRaisesRegex(RuntimeError, "process pool exists"):
                self.m.install()
            pool.process_pool.cache_clear()
            modules["torch"].__version__ = "2.14.0+cu129"
            with self.assertRaisesRegex(RuntimeError, "requires"):
                self.m.install()
            modules["torch"].__version__ = "2.14.0+cu130"
            modules["triton"].__version__ = "3.8.1"
            with self.assertRaisesRegex(RuntimeError, "requires"):
                self.m.install()
            modules["triton"].__version__ = "3.8.0"
            self.m._POLICY["autotuner_init_sha256"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "ABI differs"):
                self.m.install()


if __name__ == "__main__":
    unittest.main()


class OrderedReductionTests(unittest.TestCase):
    def test_four_actual_tiles_preserve_store_and_restored_hash(self):
        from vllm_mach.fp8 import ordered_reduction as ordered
        from vllm_mach.mxfp8.compile_choices import canonical_function_sha256
        files = ROOT / 'tests/fixtures/fp8_ordered'
        for mode in ('quality4', 'quality32', 'quality64', 'production'):
            module = load_module()
            policy = module.load_policy(mode)
            seen = set()
            for row in policy['bindings']:
                for variant in row['variants']:
                    body = variant['boundary']['function_sha256']
                    if body not in (ordered._E875, ordered._558):
                        self.assertIsNone(ordered.specification(body, row['selected_fields']))
                        continue
                    filename = 'premlp_norm.py' if body == ordered._E875 else 'layer1_norm.py'
                    source = (files / filename).read_text()
                    name = next(n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
                    changed = ordered.rewrite(source, name, variant['boundary'], row['selected_fields'])
                    after = canonical_function_sha256(changed, name)
                    self.assertEqual(ordered.original_body(after, row['selected_fields']), body)
                    original_fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
                    changed_fn = next(n for n in ast.parse(changed).body if isinstance(n, ast.FunctionDef) and n.name == name)
                    self.assertEqual(ast.dump(original_fn.body[-1]), ast.dump(changed_fn.body[-1]))
                    bad = {**variant['boundary'], 'compiler_flags': {**variant['boundary']['compiler_flags'], 'enable_fp_fusion': False}}
                    with self.assertRaises(ValueError):
                        ordered.rewrite(source, name, bad, row['selected_fields'])
                    with self.assertRaises(RuntimeError):
                        ordered.rewrite(source.replace('1e-06', '2e-06'), name, variant['boundary'], row['selected_fields'])
                    seen.add(body)
            self.assertEqual(seen, {ordered._E875, ordered._558})

    def test_work_qualified_math_files_do_not_require_runtime_imports(self):
        from vllm_mach.fp8 import ordered_reduction as ordered
        for body in (ordered._E875, ordered._558):
            with self.assertRaises(RuntimeError):
                ordered.specification(body, {'R0_BLOCK': 512})
        self.assertEqual(ordered.original_body('unrelated', {}), 'unrelated')

    def test_qualified_quant_guard_and_common_prep_mathematical_identity(self):
        import hashlib
        expected = json.loads((ROOT / 'tests/fixtures/fp8_legacy_math.json').read_text())
        from vllm_mach.mxfp8.compile_choices import canonical_function_sha256
        source = ROOT / 'src/vllm_mach'
        for filename, functions in expected['functions'].items():
            text = (source / 'fp8' / filename).read_text()
            for name, want in functions.items():
                self.assertEqual(canonical_function_sha256(text, name), want)
        self.assertEqual(hashlib.sha256((source / '_gdn_prep_kernel.py').read_bytes()).hexdigest(), expected['shared_prep_file_sha256'])
