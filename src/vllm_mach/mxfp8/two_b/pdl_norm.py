# SPDX-License-Identifier: Apache-2.0
"""Source-bound legacy local ordering for the FP32 FULL16 six-pointer norm."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

from ..compile_choices import canonical_function_sha256

SEMANTIC = 'b91191516516cd2c77a0fdee30cdbf53788bee8aedce5916a93a4b6fc5ab0451'
BEFORE = 'ef792e120ad2e23499b76d7b05b07db60fcea47ffe060df1db5ab65cdf34c88e'
AFTER = '3dfe8d71a478c7626849e805ccdea2af8fefbb8fcf2d05914a1d09eef666ac36'
CONFIG = {'kwargs': {'R0_BLOCK': 1024, 'XBLOCK': 2},
          'num_warps': 8, 'num_stages': 1, 'num_ctas': 1}
_HELPERS = Path(__file__).parents[1] / '_ordered_reduction_2b_kernel.py'


def selected(spec, rows, config):
    return (os.environ.get('VLLM_MACH_PROFILE') == 'qwen35-2b-mxfp8-champion-v1'
            and os.environ.get('VLLM_MACH_MXFP8_SSM_DTYPE', 'float32') == 'float32'
            and rows == 16 and config == CONFIG and spec['semantic_key'] == SEMANTIC)


def source(spec, rows, config):
    original = spec['source']
    if not selected(spec, rows, config):
        return original
    if (hashlib.sha256(original.encode()).hexdigest() != spec['pdl_source_sha256']
            or canonical_function_sha256(original, spec['name']) != BEFORE):
        raise RuntimeError('Six-pointer PDL norm source differs')
    policy = json.loads((Path(__file__).with_name('data') / 'ordered_reduction.json').read_text())
    helpers = _HELPERS.read_text()
    if hashlib.sha256(helpers.encode()).hexdigest() != policy['helpers_sha256']:
        raise RuntimeError('Six-pointer PDL norm helpers differ')
    tree = ast.parse(original)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == spec['name'])
    output_loop = copy.deepcopy(fn.body[-2])  # The final node launches PDL dependents.
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    for node in loop.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'tmp13':
            node.value = ast.parse('_legacy_fma(tmp9, tmp9, _tmp12)', mode='eval').body
    loop.body[:] = [n for n in loop.body if not (isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) in ('tmp10', 'tmp11'))]
    reduction = next(n for n in fn.body if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'tmp12')
    reduction.value = ast.parse('_old_local_sum(_tmp12, XBLOCK, R0_BLOCK, 8)', mode='eval').body
    if ast.dump(output_loop) != ast.dump(fn.body[-2]):
        raise RuntimeError('Six-pointer PDL norm output loop changed')
    names = {'_old_ordered_add', '_old_ordered_mul', '_old_local_sum', '_legacy_fma'}
    functions = [n for n in ast.parse(helpers).body if isinstance(n, ast.FunctionDef) and n.name in names]
    index = tree.body.index(fn)
    tree.body[index:index] = functions
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    if canonical_function_sha256(result, spec['name']) != AFTER:
        raise RuntimeError('Six-pointer PDL norm transformed identity differs')
    return result
