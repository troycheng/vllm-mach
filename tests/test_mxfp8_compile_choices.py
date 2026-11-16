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
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    fn = next((node for node in functions if node.name.startswith("triton_")), functions[0])
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
        module._POLICY = {"format_version": 1, "torch_version": "2.14.0+cu130", "triton_version": "3.8.0",
                          "bindings": bindings, "canonical_conflicts": [],
                          "autotuner_init_sha256": "fixture"}
        meta = {"signature": {"in_ptr0": "*bf16", "out_ptr0": "*bf16",
                              "xnumel": "i32", "XBLOCK": "constexpr"},
                "constants": {}, "configs": [{(0,): [["tt.divisibility", 16]]}],
                "device": types.SimpleNamespace(type="cuda", cc=120),
                "enable_fp_fusion": True, "launch_pdl": False, "disable_ftz": False}
        return bindings, meta

    def test_public_policy_has_16_construction_bindings_11_keys_and_explicit_conflicts(self):
        policy = self.module.load_policy()
        self.assertEqual(len(policy["bindings"]), 16)
        self.assertEqual(policy["format_version"], 2)
        self.assertEqual(len({row["semantic_key"] for row in policy["bindings"]}), 11)
        self.assertEqual(len(policy["canonical_conflicts"]), 3)
        self.assertTrue(all(r["construction_kind"] in ("template", "generated") for r in policy["bindings"]))
        conflict_sizes = sorted(len(r["bindings"]) for r in policy["canonical_conflicts"])
        self.assertEqual(conflict_sizes, [2, 2, 2])
        text = self.module._POLICY_FILE.read_text()
        for forbidden in ("/task", "/legacy", ".best_config", "cache_key", "source_relative_path"):
            self.assertNotIn(forbidden, text)
        for row in policy["bindings"]:
            self.assertEqual(row["semantic_key"], self.module._json_hash(row["boundary"]))
            self.assertEqual(row["boundary"]["architecture"], {"type": "cuda", "cc": 120})

    def test_actual_033_copy_source_matches_by_construction_and_exact_math(self):
        path = SOURCE_FILE.parents[3] / "tests/data/inductor033-combo-copy.py.txt"
        source = path.read_text()
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        deco = next(n for n in fn.decorator_list if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "pointwise")
        keywords = {n.arg: n.value for n in deco.keywords}
        meta = {}
        for k, v in zip(keywords["triton_meta"].keys, keywords["triton_meta"].values):
            name = ast.literal_eval(k)
            meta[name] = ({x.arg: ast.literal_eval(x.value) for x in v.keywords}
                          if name == "device" else ast.literal_eval(v))
        shape = ast.literal_eval(keywords["size_hints"])
        source = function_source(source)  # CachingAutotuner receives JITFunction.src.
        generated = self.module.select_recipe(source, shape, meta,
            kernel_alias=fn.name, kernel_site=fn.name)
        self.assertEqual(generated["construction_kind"], "generated")
        self.assertEqual(generated["selected_fields"]["XBLOCK"], 512)
        renamed = source.replace(fn.name, "triton_poi_fused_999")
        self.assertEqual(self.module.select_recipe(renamed, shape, meta,
            kernel_alias="triton_poi_fused_999", kernel_site="triton_poi_fused_999")["id"], generated["id"])
        template = self.module.select_recipe(source, shape, meta,
            kernel_alias="triton_", kernel_site="Placeholder.DESCRIPTIVE_NAME")
        self.assertEqual(template["construction_kind"], "template")
        self.assertEqual(template["selected_fields"]["XBLOCK"], 256)
        self.assertEqual(len(self.module.inspect()["seen_bindings"]), 2)
        self.assertFalse(any(k.startswith("choice_") for k in self.module.inspect()["seen_bindings"]))
        for flags in ({"enable_fp_fusion": False}, {"disable_ftz": True},
                      {"configs": [{(0,): [["tt.divisibility", 8]]}]}):
            with self.assertRaisesRegex(self.module.CompileChoiceMismatch, "boundary changed"):
                self.module.select_recipe(source, shape, {**meta, **flags}, kernel_alias=fn.name, kernel_site=fn.name)
        changed = source.replace("tl.store(out_ptr0 + (x2), tmp0, xmask)",
                                 "tl.store(out_ptr0 + (x2), tmp0 + 0.125, xmask)")
        self.assertNotEqual(changed, source)
        self.assertIsNone(self.module.select_recipe(changed, shape, meta,
            kernel_alias=fn.name, kernel_site=fn.name))
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch, "exact site"):
            self.module.select_recipe(source, shape, meta, kernel_alias=fn.name, kernel_site="wrong_site")
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch, "coverage incomplete"):
            self.module.verify_coverage()

    def test_gdn_output_pointer_update_is_one_explicit_math_preserving_rename(self):
        source = (SOURCE_FILE.parents[3] / "tests/data/inductor033-gdn-output-rename.py.txt").read_text()
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        new = self.module.recipe_from_source(source, fn.name, {}, "fixture")
        self.assertEqual(new["semantic_key"], "62f2fdf17bbc0f38ec0752a423a9490c9f47e53a050cfad7da869fdfc1c1b185")
        old = self.module.recipe_from_source(source.replace("out_ptr2", "out_ptr1"), fn.name, {}, "fixture")
        self.assertEqual(old["semantic_key"], "000bcbaee5e2ee61dc880753956da52de3253e0fe53ad9f66701a1a6d6f49e50")
        self.assertEqual(fn.args.args[3].arg, "out_ptr2")
        rows = [r for r in self.module.load_policy()["bindings"] if r["semantic_key"] == new["semantic_key"]]
        self.assertEqual({r["construction_kind"]:r["selected_fields"]["XBLOCK"] for r in rows},
                         {"template":8,"generated":32})

    def test_actual_residual_rms_keeps_gdn_tree_despite_kernel_renumbering(self):
        data = SOURCE_FILE.parents[3] / "tests/data"
        norm = (data/"inductor033-residual-rms.py.txt").read_text()
        gdn = (data/"inductor033-gdn-output-rename.py.txt").read_text()
        fn = next(n for n in ast.parse(norm).body if isinstance(n, ast.FunctionDef))
        deco = next(n for n in fn.decorator_list if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "reduction")
        keywords = {n.arg:n.value for n in deco.keywords}; meta = {}
        for k,v in zip(keywords["triton_meta"].keys, keywords["triton_meta"].values):
            name = ast.literal_eval(k)
            meta[name] = ({x.arg:ast.literal_eval(x.value) for x in v.keywords}
                          if name == "device" else ast.literal_eval(v))
        shape = ast.literal_eval(keywords["size_hints"])
        choose = lambda **kwargs:self.module.select_recipe(function_source(norm),shape,meta,
            kernel_alias=fn.name,kernel_site=fn.name,**kwargs)
        # Exact live module code constants identify the GDN partition. The new
        # norm_1 must use GDN's old4096 tree, not attention's old1024 tree.
        program = "gdn_source = " + repr(gdn) + "\nnorm_source = " + repr(norm) + "\nanswer = choose()\n"
        scope = {"choose":choose};exec(compile(program,"<generated-wrapper-contract>","exec"),scope)
        row = scope["answer"]
        self.assertEqual(row["selected_fields"]["R0_BLOCK"],4096)
        self.assertEqual(row["selected_fields"]["num_warps"],16)
        anchors = {r["context_semantic_key"] for r in self.module.load_policy()["bindings"]
                   if "context_semantic_key" in r}
        attention = next(key for key in anchors if key.startswith("9f65"))
        self.assertEqual(choose(kernel_context={attention})["selected_fields"]["R0_BLOCK"],1024)
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch,"exact site"):
            choose(kernel_context=anchors)
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch,"exact site"):
            choose(kernel_context=set())
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch,"exact site"):
            choose()
        keys = self.module._code_semantic_keys((gdn,norm))
        self.assertEqual(len(keys),2)
        self.assertIn(row["semantic_key"],keys)
        self.assertIn(row["context_semantic_key"],keys)

    def test_identical_norm_modules_keep_separate_gdn_and_attention_cached_tuners(self):
        data = SOURCE_FILE.parents[3] / "tests/data"
        norm = (data/"inductor033-residual-rms.py.txt").read_text()
        anchors = [(data/"inductor033-gdn-output-rename.py.txt").read_text(),
                   (data/"inductor033-attention-sigmoid.py.txt").read_text()]
        fn = next(n for n in ast.parse(norm).body if isinstance(n, ast.FunctionDef))
        module = self.module
        class Compiler:
            def __init__(self):self.cache = {};self.constructions = []
            def triton(self, name, source, *, device_str):
                if source in self.cache:return self.cache[source]
                tree = ast.parse(source)
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                deco = next(n for n in function.decorator_list if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute) and n.func.attr == "reduction")
                keywords = {n.arg:n.value for n in deco.keywords}
                imeta = ast.literal_eval(keywords["inductor_meta"])
                meta = {}
                for k,v in zip(keywords["triton_meta"].keys, keywords["triton_meta"].values):
                    key = ast.literal_eval(k)
                    meta[key] = ({x.arg:ast.literal_eval(x.value) for x in v.keywords}
                                 if key == "device" else ast.literal_eval(v))
                row = module.select_recipe(function_source(source),ast.literal_eval(keywords["size_hints"]),meta,
                    kernel_alias=name,kernel_site=name,kernel_context={imeta["mach_compile_context"]})
                self.cache[source] = row;self.constructions.append((source, row, device_str))
                return row
        Compiler.triton = module._contextual_triton(Compiler.triton)
        compiler = Compiler();results = []
        for anchor in anchors:
            # Execute only a source-string container + compiler call. Neither
            # captured Triton source nor Torch is imported/executed.
            program = ("anchor_source = " + repr(anchor) + "\nnorm_source = " + repr(norm)
                       + "\nanswer = compiler.triton(name,norm_source,device_str='cuda')\n")
            scope = {"compiler":compiler,"name":fn.name}
            exec(compile(program,"<generated-cache-contract>","exec"),scope)
            results.append(scope["answer"])
        self.assertIsNot(results[0],results[1])
        self.assertEqual([r["selected_fields"]["R0_BLOCK"] for r in results],[4096,1024])
        self.assertEqual([r["selected_fields"]["num_warps"] for r in results],[16,8])
        self.assertEqual(len(compiler.cache),2)
        before = module.recipe_from_source(norm,fn.name,{},"fixture")["boundary"]
        for source,row,device in compiler.constructions:
            expected = {**before, "function_sha256": row["ordered_reduction"]["function_sha256"]} if "ordered_reduction" in row else before
            self.assertEqual(module.recipe_from_source(source,fn.name,{},"fixture")["boundary"],expected)
            self.assertEqual(module.canonical_function_sha256(source,fn.name),expected["function_sha256"])
            self.assertEqual(device,"cuda")
            self.assertIn("'mach_compile_context':",source)

    def test_actual_quality_boundaries_preserve_each_contexts_old_launch(self):
        fixtures = json.loads((SOURCE_FILE.parents[3]/"tests/data/inductor033-quality-boundaries.json").read_text())
        expected = {
            "quality32": {"norm":{"gdn":(1024,8),"attention":(4096,16)},
                          "swiglu":{"gdn":(1024,4),"attention":(512,8)},
                          "copy":{"gdn":(512,8),"attention":(1024,4),"embedding":(1024,4)}},
            "quality64": {"norm":{"gdn":(1024,8),"attention":(1024,8)},
                          "swiglu":{"gdn":(1024,4),"attention":(1024,4)},
                          "copy":{"gdn":(512,8),"attention":(1024,4),"embedding":(1024,4)}}}
        class Compiler:
            def __init__(self,module):self.module=module;self.cache={}
            def triton(self,name,source,**kwargs):
                if source in self.cache:return self.cache[source]
                fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == name)
                deco=next(n for n in fn.decorator_list if isinstance(n,ast.Call)
                          and isinstance(n.func,ast.Attribute) and any(k.arg=="inductor_meta" for k in n.keywords))
                keywords={k.arg:k.value for k in deco.keywords};imeta=ast.literal_eval(keywords["inductor_meta"]);meta={}
                for k,v in zip(keywords["triton_meta"].keys,keywords["triton_meta"].values):
                    key=ast.literal_eval(k)
                    meta[key]=({x.arg:ast.literal_eval(x.value) for x in v.keywords}
                               if key=="device" else ast.literal_eval(v))
                row=self.module.select_recipe(function_source(source),ast.literal_eval(keywords["size_hints"]),meta,
                    kernel_alias=name,kernel_site=imeta["kernel_name"],
                    kernel_context={imeta["mach_compile_context"]} if "mach_compile_context" in imeta else None)
                self.cache[source]=row;return row
        for mode, families in expected.items():
            with patch.dict(os.environ,{"VLLM_MACH_MXFP8_MODE":"quality","VLLM_MACH_MXFP8_QUALITY_ROWS":mode[-2:]}):
                module=load_module();data=fixtures[mode]
                for family,roles in families.items():
                    compiler=Compiler(module)
                    compiler.triton=module._contextual_triton(Compiler.triton).__get__(compiler,Compiler)
                    source=data[family+"_source"]
                    name=next(n.name for n in ast.parse(source).body if isinstance(n,ast.FunctionDef))
                    for role,(block,warps) in roles.items():
                        with self.subTest(mode=mode,family=family,role=role):
                            program=("anchor_source="+repr(data[role+"_anchor_source"])+"\nsource="+repr(source)
                                     +"\nanswer=compiler.triton(name,source,device_str='cuda')\n")
                            scope={"compiler":compiler,"name":name}
                            exec(compile(program,"<quality-generated-context>","exec"),scope)
                            fields=scope["answer"]["selected_fields"]
                            self.assertEqual(fields.get("R0_BLOCK",fields["XBLOCK"]),block)
                            self.assertEqual(fields["num_warps"],warps)
                    separate=len(roles) if (family=="copy" or mode=="quality32") else 1
                    self.assertEqual(len(compiler.cache),separate)
                # Template construction has no enclosing graph anchor. It must
                # retain its own1024/w4 choice, never acquire generated tags.
                source=data["copy_source"]
                name=next(n.name for n in ast.parse(source).body if isinstance(n,ast.FunctionDef))
                source=source.replace(name,"triton_").replace("'kernel_name': 'triton_'","'kernel_name': 'Placeholder.DESCRIPTIVE_NAME'")
                compiler=Compiler(module);compiler.triton=module._contextual_triton(Compiler.triton).__get__(compiler,Compiler)
                row=compiler.triton("triton_",source,device_str="cuda")
                self.assertEqual(row["construction_kind"],"template")
                self.assertEqual(row["selected_fields"]["XBLOCK"],1024)
                self.assertEqual(row["selected_fields"]["num_warps"],4)
                self.assertTrue(all("mach_compile_context" not in source for source in compiler.cache))
                gdn=data["gdn_anchor_source"];name=next(n.name for n in ast.parse(gdn).body if isinstance(n,ast.FunctionDef))
                restored=module.recipe_from_source(gdn.replace("out_ptr2","out_ptr1"),name,{},"old")
                self.assertEqual(restored["semantic_key"],
                    "47ceb13fbbc6926d4b80435981cccc2fd3f9a4c1a2b8115d48c0ff2781fe5b2d" if mode=="quality32" else
                    "1050ac67757dd5a7afa8aa25e52f90a9e44e69f651f1d2a4ee1d9b162b4f754b")

    def test_same_construction_different_reduction_config_still_requires_exact_site(self):
        bindings, meta = self.fixture()
        self.module._POLICY["format_version"] = 2
        for row in bindings: row["construction_kind"] = "generated"
        bindings[1]["selected_fields"] = {"XBLOCK": 256, "num_warps": 4,
                                          "num_stages": 1, "found_by_coordesc": False}
        source = function_source(generated_source())
        with self.assertRaisesRegex(self.module.CompileChoiceMismatch, "exact site"):
            self.module.select_recipe(source, {"x": 65536}, meta,
                kernel_alias="triton_poi_fused_fixture_999", kernel_site="triton_poi_fused_fixture_999")
        row = self.module.select_recipe(source, {"x": 65536}, meta,
            kernel_alias=bindings[1]["kernel_alias"], kernel_site=bindings[1]["kernel_site"])
        self.assertEqual(row["id"], bindings[1]["id"])

    def test_target_compiler_versions_reject_old_patch_cuda_and_triton_drift(self):
        policy = self.module.load_policy()
        self.assertEqual(policy["torch_version"], "2.14.0+cu130")
        self.assertEqual(policy["triton_version"], "3.8.0")
        self.module._require_compiler_versions(policy, "2.14.0+cu130", "3.8.0")
        for torch_version, triton_version in (
                ("2.13.0+cu130", "3.8.0"), ("2.14.1+cu130", "3.8.0"),
                ("2.14.0+cu129", "3.8.0"), ("2.14.0", "3.8.0"),
                ("2.14.0+cu130", "3.7.1"), ("2.14.0+cu130", "3.8.1")):
            with self.subTest(torch=torch_version, triton=triton_version):
                with self.assertRaisesRegex(RuntimeError, "Compile policy requires"):
                    self.module._require_compiler_versions(policy, torch_version, triton_version)

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
        torch.__version__ = "2.14.0+cu130"
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
        triton.__version__ = "3.8.0"
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
        for rows, count in ((32, 19), (64, 16)):
            module = load_module()
            with patch.dict(os.environ, {"VLLM_MACH_MXFP8_MODE": "quality",
                                         "VLLM_MACH_MXFP8_QUALITY_ROWS": str(rows)}):
                policy = module.load_policy()
                self.assertEqual(len(policy["bindings"]), count)
                self.assertEqual(policy["profile_mode"], f"quality{rows}")
                self.assertEqual(module.inspect()["policy_mode"], f"quality{rows}")
                self.assertEqual({row["boundary"]["architecture"]["cc"]
                                  for row in policy["bindings"]}, {120})
                self.assertEqual(policy["format_version"], 2)
                self.assertTrue(policy["baseline_inventory_sha256"])
                self.assertTrue(all(row.get("observed_source_sha256") and row["construction_kind"]
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
            raise unittest.SkipTest("Real Torch 2.14/Triton 3.8.0 runtime required")
        if torch.__version__ != "2.14.0+cu130" or triton.__version__ != "3.8.0":
            raise unittest.SkipTest("Test binds the locked Torch 2.14/Triton 3.8.0 ABI")
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
        module._POLICY = {"format_version": 1, "bindings": [row], "canonical_conflicts": []}
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
