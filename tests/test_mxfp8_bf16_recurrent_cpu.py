"""Qualified BF16 layout admission, original fallback and exact kernel/helper ASTs."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
GDN = ROOT / 'src/vllm_mach/mxfp8/gdn'
FIXTURE = json.loads((ROOT / 'tests/fixtures/mxfp8_bf16_recurrent_order.json').read_text())


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def specialize(fn, enabled):
    class Select(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == 'legacy_bf16':
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == 'legacy_bf16':
                return [self.visit(n) for n in (node.body if enabled else node.orelse)]
            return self.generic_visit(node)
    return Select().visit(copy.deepcopy(fn))


def test_qualified_bf16_math_and_fallback_restore_entire_original_kernel():
    for file, proof in FIXTURE['kernels'].items():
        fn = function(GDN / file, '_ordered_decode')
        assert digest(specialize(fn, True)) == proof['qualified_candidate_AST']
        assert digest(specialize(fn, False)) == proof['original_AST']


def test_bf16_admission_excludes_fp32_and_unqualified_layouts():
    for file in FIXTURE['kernels']:
        fn = function(GDN / file, '_ordered_decode')
        gate = next(n.value for n in fn.body if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == 'legacy_bf16')
        code = compile(ast.Expression(gate), file, 'eval')
        def enabled(dtype='bf16', **changes):
            values = dict(H=16, HV=16 if file.startswith('../two_b/') else 32, K=128, V=128, BV=32, NV=4, W=4,
                          FULL=False, STATE_ROWS=1, USE_PDL=False, warps=4)
            values.update(changes)
            lang = NS(bfloat16='bf16', num_warps=lambda: values['warps'])
            return eval(code, {'__builtins__': {}}, dict(values,
                        base=NS(dtype=NS(element_ty=dtype)), gl=lang, tl=lang))
        assert enabled()
        assert not enabled('fp32') and not enabled('fp16')
        for field, value in [('H',8), ('K',64), ('V',64), ('BV',64), ('NV',2),
                             ('W',8)]:
            assert not enabled(**{field:value})
        if file == 'bf16_small_gluon.py':
            assert enabled(HV=16)
            assert enabled(FULL=True,warps=8)
            assert enabled(FULL=False,warps=2)
            for change in [dict(HV=24),dict(FULL=True,warps=4),dict(FULL=True,warps=2),dict(STATE_ROWS=4),dict(FULL=False,warps=8),dict(USE_PDL=True)]:
                assert not enabled(**change)
        else:
            assert not enabled(HV=32 if file.startswith('../two_b/') else 16)
            if not file.startswith('../two_b/'):assert not enabled(USE_PDL=True)
            launch = function(GDN / file, 'decode')
            call = next(n for n in ast.walk(launch) if isinstance(n,ast.Call)
                        and isinstance(n.func,ast.Subscript)
                        and isinstance(n.func.value,ast.Name) and n.func.value.id == '_ordered_decode')
            assert next(k.value.value for k in call.keywords if k.arg == 'num_warps') == 1


def test_qualified_local8_helpers_and_original_local4_functions_are_unchanged():
    for file, expected in FIXTURE['helpers'].items():
        for name, expected_hash in expected.items():
            assert digest(function(GDN / file, name)) == expected_hash
    local4 = json.loads((ROOT / 'tests/fixtures/mxfp8_2b_recurrent_order.json').read_text())
    for name, expected_hash in local4['qualified_helper_functions'].items():
        assert digest(function(GDN / 'legacy_reduction.py', name)) == expected_hash
