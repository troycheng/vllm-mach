"""Finite 2B arithmetic/selection contracts; no GPU or installed vLLM needed."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vllm_mach import gdn_prep
from vllm_mach.mxfp8.compile_choices import canonical_function_sha256, recipe_from_source
from vllm_mach.mxfp8.two_b import attention, ordered_reduction

CASES = json.loads((Path(__file__).parent / 'fixtures/mxfp8_2b_ordered_sources.json').read_text())


def test_selected_mxfp8_actual_compiler_lifecycle(tmp_path):
    # cpu_suites.json admits this sentinel only for an explicitly installed
    # MXFP8 profile. Missing imports, wrong stack or source guards must fail.
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'VLLM_PLUGINS': '',
           'PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tools'),
                                        str(ROOT / 'tools/quantization')])}
    script = Path(__file__).parent / 'fixtures/mxfp8_2b_compiler_lifecycle.py'
    subprocess.run([sys.executable, str(script), str(tmp_path / 'actual')],
                   env=env, check=True, timeout=180)
    result = json.loads((tmp_path / 'actual/PRODUCT_LIFECYCLE_CPU.json').read_text())
    assert result['passed'] and not result['GPU_initialized']


def test_selected_mxfp8_4b_keeps_actual_stock_prep_callable():
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'VLLM_PLUGINS': '',
           'VLLM_MACH_PROFILE': 'qwen35-4b-mxfp8-champion-v1',
           'PYTHONPATH': str(ROOT / 'src')}
    subprocess.run([sys.executable, '-c', '''
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('CPU binding check attempted CUDA initialization')
torch.cuda._lazy_init = forbidden
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
assert gdn.fused_post_conv_prep is fused_post_conv_prep
assert not torch.cuda.is_initialized()
'''], env=env, check=True, timeout=120)


class Ordered2BCPU(unittest.TestCase):
    def test_finite_norm_transforms_and_all_actual_hint_variants(self):
        policy = ordered_reduction.policy()
        for case in CASES:
            with self.subTest(body=case['before']):
                changed = ordered_reduction.rewrite(case['source'], case['name'])
                self.assertEqual(canonical_function_sha256(changed, case['name']), case['after'])
                self.assertEqual(ordered_reduction.rewrite(changed, case['name']), changed)
                before = recipe_from_source(case['source'], case['name'], {}, case['name'])['boundary']
                after = recipe_from_source(changed, case['name'], {}, case['name'])['boundary']
                self.assertEqual(after, {**before, 'function_sha256': case['after']})
                # Hints change only the declared launch geometry; all three
                # entries were read from actual production/M32/M64 AOT source.
                hints = policy['targets'][case['before']]['size_hints']
                self.assertEqual(len(hints), 3)
                self.assertEqual({h['x'] for h in hints},
                                 {32768, 131072, 262144} if before['size_hints']['r0_'] == 128 else {2048, 8192, 16384})
                for variant in hints:
                    tree = ast.parse(case['source'])
                    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == case['name'])
                    decorator = next(n for n in fn.decorator_list if isinstance(n, ast.Call) and any(k.arg == 'size_hints' for k in n.keywords))
                    next(k for k in decorator.keywords if k.arg == 'size_hints').value = ast.parse(repr(variant), mode='eval').body
                    source = ast.unparse(tree)
                    result = ordered_reduction.rewrite(source, case['name'])
                    self.assertEqual(canonical_function_sha256(result, case['name']), case['after'])

    def test_math_flags_helpers_and_unknown_descriptors_reject(self):
        for case in CASES:
            for bad in [case['source'].replace('1e-06', '2e-06'),
                        case['source'].replace("'enable_fp_fusion': True", "'enable_fp_fusion': False"),
                        case['source'].replace('8192', '8193').replace('131072', '131073')]:
                with self.subTest(body=case['before'], bad=bad[:24]):
                    self.assertNotEqual(bad, case['source'])
                    with self.assertRaises((RuntimeError, AssertionError, ValueError)):
                        ordered_reduction.rewrite(bad, case['name'])
            changed = ordered_reduction.rewrite(case['source'], case['name'])
            with self.assertRaises((RuntimeError, AssertionError)):
                ordered_reduction.rewrite(changed.replace('add.rn.f32', 'sub.rn.f32', 1), case['name'])

    def test_attention_and_prep_only_select_explicit_profile_geometry(self):
        name = 'qwen35-2b-mxfp8-champion-v1'
        for profile in [None, '', 'native-mxfp6-sm120-v033', 'qwen35-4b-mxfp8-champion-v1', name]:
            self.assertEqual(attention.selected(profile, 16, 256, 256, True, False), profile == name)
            self.assertEqual(gdn_prep.selected(profile, heads=16, value_heads=16, head_dim=128,
                                              value_dim=128, bf16=True, apply_l2norm=True,
                                              output_g_exp=False), profile == name)
        for args in [(64, 256, 256, True, False), (16, 128, 128, True, False),
                     (16, 256, 256, False, False), (16, 256, 256, True, True)]:
            self.assertFalse(attention.selected(name, *args))
        for field, value in [('value_heads', 32), ('heads', 8), ('head_dim', 64),
                             ('value_dim', 64), ('bf16', False), ('apply_l2norm', False),
                             ('output_g_exp', True)]:
            kw = dict(heads=16, value_heads=16, head_dim=128, value_dim=128, bf16=True,
                      apply_l2norm=True, output_g_exp=False);kw[field] = value
            self.assertFalse(gdn_prep.selected(name, **kw))

    def test_packaged_math_identity_and_4b_helper_is_unchanged(self):
        package = ROOT / 'src/vllm_mach'
        # Exact qualified component source, including all epsilon/gating/store
        # arithmetic; no work path or source-file mutation is needed at runtime.
        self.assertEqual(hashlib.sha256((package / '_gdn_prep_kernel.py').read_bytes()).hexdigest(),
                         'd574290a6b3dea264fe7a9f257be30647df9877038284e30c601b9add78dc30c')
        self.assertEqual(hashlib.sha256((package / 'mxfp8/_ordered_reduction_2b_kernel.py').read_bytes()).hexdigest(),
                         ordered_reduction.policy()['helpers_sha256'])
        # Existing 4B specs remain a different helper file and unchanged trees.
        old = package / 'mxfp8/_ordered_reduction_kernel.py'
        self.assertNotEqual(old.read_bytes(), (package / 'mxfp8/_ordered_reduction_2b_kernel.py').read_bytes())
        four_b = json.loads((package / 'mxfp8/data/compile_choices_quality32.json').read_text())
        self.assertTrue(any('ordered_reduction' in r for r in four_b['bindings']))
        for row in four_b['bindings']:
            if 'ordered_reduction' in row:
                self.assertEqual(row['ordered_reduction']['helper_sha256'], hashlib.sha256(old.read_bytes()).hexdigest())
        tree = ast.parse((package / 'mxfp8/two_b/_attention_reduction_kernel.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reduce_segments_explicit_old_lane')
        self.assertTrue(any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'tl.inline_asm_elementwise' for n in ast.walk(fn)))


if __name__ == '__main__':
    unittest.main()
