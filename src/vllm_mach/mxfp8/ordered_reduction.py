# SPDX-License-Identifier: Apache-2.0
"""Finite source-bound SM120 legacy FP32 norm ordering; CPU-only AST helpers."""
from __future__ import annotations
import ast
import copy
import hashlib
import re
from pathlib import Path

_KERNEL = Path(__file__).with_name('_ordered_reduction_kernel.py')
_FIELDS = {'version', 'group', 'chunk_accumulation', 'function_sha256', 'helper_sha256'}


def helper_source():
    return _KERNEL.read_text()


def helper_sha256():
    return hashlib.sha256(_KERNEL.read_bytes()).hexdigest()


def active(spec):
    if spec.get('version') == 2:
        from vllm_mach.gdn_prep import four_b_enabled
        return four_b_enabled()
    return True


def validate(spec, boundary, selected):
    if spec.get('version') == 2:
        if (set(spec) != {'version', 'kind', 'function_sha256', 'helper_sha256'}
                or spec['kind'] != 'gdn128_bf16'
                or spec['function_sha256'] != 'dde78dd0e0fbf58b105ba9f8d64d749c0901050bdd71ba86bd29305adf1b5ff8'
                or spec['helper_sha256'] != hashlib.sha256(Path(__file__).with_name('_ordered_reduction_gdn_kernel.py').read_bytes()).hexdigest()
                or boundary['function_sha256'] != '85920b6fffb03e34d3c4b654155b8810330577e6609294e1044cb7e974992ce9'
                or boundary['size_hints'] not in ({'x': 65536, 'r0_': 128}, {'x': 262144, 'r0_': 128}, {'x': 524288, 'r0_': 128})
                or selected != {'XBLOCK': 32, 'num_warps': 2, 'num_stages': 1, 'found_by_coordesc': False}
                or boundary['architecture'] != {'type': 'cuda', 'cc': 120}
                or boundary['compiler_flags'] != {'enable_fp_fusion': True, 'disable_ftz': False, 'launch_pdl': False, 'native_matmul': False}):
            raise ValueError('Unsupported BF16 GDN output norm boundary')
        return
    if set(spec) != _FIELDS or type(spec['version']) is not int or spec['version'] != 1:
        raise ValueError('Unknown ordered norm source contract')
    group = spec['group']
    if group not in (4, 8) or spec['chunk_accumulation'] not in (('fma', 'separate') if group == 4 else ('original',)):
        raise ValueError('Unsupported ordered norm local/FMA tree')
    if (selected['XBLOCK'], selected['R0_BLOCK'], selected['num_warps']) != (1, 1024 if group == 4 else 4096, 8 if group == 4 else 16):
        raise ValueError('Ordered norm tile differs')
    if boundary['architecture'] != {'type': 'cuda', 'cc': 120} or any(
            value != '*bf16' for value in boundary['signature'].values() if isinstance(value, str) and value.startswith('*')):
        raise ValueError('Ordered norm architecture/dtype differs')
    if boundary['compiler_flags'] != {'enable_fp_fusion': True, 'disable_ftz': False, 'launch_pdl': False, 'native_matmul': False}:
        raise ValueError('Ordered norm compiler flags differ')
    if spec['helper_sha256'] != helper_sha256():
        raise ValueError('Ordered norm helper source differs')
    if not isinstance(spec['function_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', spec['function_sha256']):
        raise ValueError('Invalid ordered norm function identity')


def transform(source, kernel_name, spec):
    """Replace only approved accumulator/local sum, preserving the output loop."""
    if spec.get('version') == 2:
        return _transform_gdn(source, kernel_name)
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == kernel_name)
    original = copy.deepcopy(fn)
    sums = [node for node in ast.walk(fn) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Call) and ast.unparse(node.value.value.func) == 'tl.sum']
    if len(sums) != 1 or ast.literal_eval(sums[0].value.value.args[1]) != 1:
        raise ValueError('Expected one row norm reduction')
    reduction = sums[0]
    accumulator = reduction.value.value.args[0]
    if not isinstance(accumulator, ast.Name):
        raise ValueError('Unexpected norm accumulator')
    if spec['chunk_accumulation'] != 'original':
        loop = next(node for node in fn.body if isinstance(node, ast.For))
        assignments = {node.targets[0].id: node for node in loop.body
                       if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
        updates = [node for node in assignments.values() if isinstance(node.value, ast.BinOp)
                   and isinstance(node.value.op, ast.Add) and isinstance(node.value.left, ast.Name)
                   and node.value.left.id == accumulator.id]
        if len(updates) != 1 or not isinstance(updates[0].value.right, ast.Name):
            raise ValueError('Unexpected norm accumulator update')
        update = updates[0]
        broadcast = assignments[update.value.right.id].value
        if not isinstance(broadcast, ast.Call) or ast.unparse(broadcast.func) != 'tl.broadcast_to':
            raise ValueError('Unexpected squared norm broadcast')
        square = assignments[broadcast.args[0].id].value
        if not isinstance(square, ast.BinOp) or not isinstance(square.op, ast.Mult) or ast.dump(square.left) != ast.dump(square.right):
            raise ValueError('Expected exact square before norm accumulation')
        if spec['chunk_accumulation'] == 'fma':
            update.value = ast.Call(func=ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='fma',ctx=ast.Load()),
                                    args=[copy.deepcopy(square.left),copy.deepcopy(square.right),copy.deepcopy(accumulator)],keywords=[])
        else:
            assignments[broadcast.args[0].id].value = ast.Call(func=ast.Name(id='_old_ordered_mul',ctx=ast.Load()),
                args=[copy.deepcopy(square.left),copy.deepcopy(square.right)],keywords=[])
            update.value = ast.Call(func=ast.Name(id='_old_ordered_add',ctx=ast.Load()),
                args=[copy.deepcopy(accumulator),copy.deepcopy(update.value.right)],keywords=[])
    reduction.value = ast.Call(func=ast.Name(id='_old_local_sum',ctx=ast.Load()),
        args=[copy.deepcopy(accumulator),ast.Name(id='XBLOCK',ctx=ast.Load()),ast.Name(id='R0_BLOCK',ctx=ast.Load()),ast.Constant(spec['group'])],keywords=[])
    if ast.dump(original.body[-1],include_attributes=False) != ast.dump(fn.body[-1],include_attributes=False):
        raise ValueError('Ordered norm changed normalization/output arithmetic')
    helpers = [node for node in ast.parse(helper_source()).body if isinstance(node, ast.FunctionDef)]
    names = {node.name for node in helpers}
    if any(isinstance(node,ast.FunctionDef) and node.name in names for node in tree.body):
        raise ValueError('Ordered norm helper already exists')
    index = tree.body.index(fn)
    tree.body[index:index] = helpers
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def validate_globals(fn, canonical_hash, spec=None):
    """The transformed target hash also binds the two referenced JIT helpers."""
    globals_ = fn.fn.__globals__
    helpers = (Path(__file__).with_name('_ordered_reduction_gdn_kernel.py').read_text()
               if spec and spec.get('version') == 2 else helper_source())
    for node in ast.parse(helpers).body:
        if isinstance(node, ast.FunctionDef):
            actual = globals_.get(node.name)
            if actual is None or canonical_hash(actual.src) != canonical_hash(ast.unparse(node)):
                raise RuntimeError('Ordered norm JIT helper differs: ' + node.name)


def transform_2b(source, name, spec):
    from vllm_mach.mxfp8.compile_choices import canonical_function_sha256
    c = {'targets': spec}
    before_hash = canonical_function_sha256(source, name)
    assert before_hash in spec
    tree = ast.parse(source)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    before = copy.deepcopy(fn)
    if before_hash == '89c262027f0c0ce34bfeaf0b88543363412f53e3c5443fefefe6c38594fab08d':
        assignments = {n.targets[0].id: n for n in fn.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        assert ast.unparse(assignments['tmp6'].value) == 'tl.sum(tmp5, 1)[:, None].to(tl.float32)'
        assert ast.unparse(assignments['tmp2'].value) == 'tmp1 * tmp1'
        assignments['tmp6'].value = ast.parse('_legacy_gdn_square_sum(tmp1, xmask, XBLOCK).to(tl.float32)', mode='eval').body
    else:
        reductions = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Call) and (ast.unparse(n.value.value.func) == 'tl.sum')]
        assert len(reductions) == 1
        reduction = reductions[0]
        call = reduction.value.value
        assert ast.literal_eval(call.args[1]) == 1 and isinstance(call.args[0], ast.Name)
        acc = call.args[0]
        loop = next((n for n in fn.body if isinstance(n, ast.For)))
        assigns = {n.targets[0].id: n for n in loop.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        updates = [n for n in assigns.values() if isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Add) and isinstance(n.value.left, ast.Name) and (n.value.left.id == acc.id)]
        assert len(updates) == 1
        update = updates[0]
        broadcast = assigns[update.value.right.id].value
        assert ast.unparse(broadcast.func) == 'tl.broadcast_to'
        square = assigns[broadcast.args[0].id].value
        assert isinstance(square, ast.BinOp) and isinstance(square.op, ast.Mult) and (ast.dump(square.left) == ast.dump(square.right))
        update.value = ast.Call(func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()), attr='fma', ctx=ast.Load()), args=[copy.deepcopy(square.left), copy.deepcopy(square.right), copy.deepcopy(acc)], keywords=[])
        group = 4 if before_hash == '3e6a42e2b0987dce312dc19c4fa55b2bb483df9802bc33e04b21a6c47b0c6d97' else 8
        reduction.value = ast.parse('_old_local_sum(' + acc.id + ', XBLOCK, R0_BLOCK, ' + str(group) + ')', mode='eval').body
    assert ast.dump(before.body[-1]) == ast.dump(fn.body[-1])
    helpers = [n for n in ast.parse(Path(__file__).with_name('_ordered_reduction_2b_kernel.py').read_text()).body if isinstance(n, ast.FunctionDef)]
    assert not set((n.name for n in helpers)) & set((n.name for n in tree.body if isinstance(n, ast.FunctionDef)))
    tree.body[tree.body.index(fn):tree.body.index(fn)] = helpers
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    assert canonical_function_sha256(result, name) == c['targets'][before_hash]['after_sha256']
    return result


def _transform_gdn(source, kernel_name):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kernel_name)
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'tmp6' for t in n.targets))
    if ast.unparse(assignment.value) != 'tl.sum(tmp5, 1)[:, None].to(tl.float32)':
        raise ValueError('GDN norm row sum differs')
    old = copy.deepcopy(assignment)
    changed = copy.deepcopy(assignment)
    changed.value = ast.parse('old_norm128_sum(tmp1, XBLOCK, R0_BLOCK)', mode='eval').body
    fn.body[fn.body.index(assignment)] = ast.If(
        test=ast.parse('XBLOCK == 32', mode='eval').body, body=[changed], orelse=[old])
    helpers = [n for n in ast.parse(Path(__file__).with_name('_ordered_reduction_gdn_kernel.py').read_text()).body
               if isinstance(n, ast.FunctionDef)]
    if {n.name for n in helpers} & {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}:
        raise ValueError('GDN norm helper name collision')
    index = tree.body.index(fn)
    tree.body[index:index] = helpers
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
