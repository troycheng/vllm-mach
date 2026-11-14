# SPDX-License-Identifier: Apache-2.0
"""Rebuild finite RMS bodies at the existing FP8 constructor boundary."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import tempfile

from ..mxfp8 import ordered_reduction as arithmetic
from ..mxfp8.compile_choices import canonical_function_sha256

_HELPERS_SHA = 'f5f648467910ffbda413d8e054ec8e2479374f2a8014857b90ffe6352f151c81'
_E875 = 'e875b0b81961dbe40ad7195d00887558772d0f2d4b803dfa59dbc7800410ed5b'
_558 = '558ac2424583841268b736426852741fe87128e1c938fa4dad3625c3d4e137ac'
_EMBED = '3e6a42e2b0987dce312dc19c4fa55b2bb483df9802bc33e04b21a6c47b0c6d97'
# Both tile orders were replayed against their actual selected 0.29 PTX.
_AFTER = {
    (_EMBED, 1024): '4480da6f4c73b10ec6a6ba9e7b2d555ae8bc9a0ce2b04bd9b02566ac4cd855ec',
    (_E875, 1024): '727bce2d92acfe174f4775c330c5126c56a9109891bd35d85b34d64d96516169',
    (_558, 4096): 'fe0038bef367024b1d6ec6c3fa52c38f3b43f21ff7486dda727819dd201e44b1',
    (_E875, 4096): '48aa89937e6fd1490bfb9d7689062dfaebc017ad9148cb9e1dffe4e5b5d0a8c2',
    (_558, 1024): '9013e3e9ff24c707c5a29dbc9163eb05f1e149085f380f33bda8f8bfef639e6e',
}


def specification(body, selected):
    if body not in (_E875, _558, _EMBED):
        return None
    tile = selected['R0_BLOCK']
    after = _AFTER.get((body, tile))
    if after is None:
        raise RuntimeError('Unqualified block-FP8 RMS tile/source')
    expected = dict(XBLOCK=1, R0_BLOCK=tile, num_warps=8 if tile == 1024 else 16,
                    num_stages=1, found_by_coordesc=False)
    if selected != expected:
        raise RuntimeError('Block-FP8 ordered RMS selected configuration changed')
    return dict(version=1, group=4 if tile == 1024 else 8,
                chunk_accumulation='fma' if tile == 1024 else 'original',
                function_sha256=after, helper_sha256=_HELPERS_SHA)


def original_body(body, selected):
    return next((original for (original, tile), after in _AFTER.items()
                 if tile == selected.get('R0_BLOCK') and body == after), body)


def rewrite(source, name, boundary, selected):
    body = canonical_function_sha256(source, name)
    if body != boundary['function_sha256']:
        raise RuntimeError('Block-FP8 RMS source/descriptor differs')
    spec = specification(body, selected)
    if spec is None:
        return source
    if body == _EMBED:
        # This gather has integer IDs; do not widen the shared BF16-only validator.
        signature = dict(in_ptr0='*i32', in_ptr1='*bf16', in_ptr2='*bf16',
                         out_ptr0='*bf16', out_ptr2='*bf16', xnumel='i32',
                         r0_numel='i32', XBLOCK='constexpr', R0_BLOCK='constexpr')
        if (boundary['signature'] != signature or boundary['constants'] != {}
                or boundary['size_hints'] not in [dict(x=x, r0_=4096) for x in (2048, 8192, 16384)]
                or boundary['architecture'] != dict(type='cuda', cc=120)
                or boundary['compiler_flags'] != dict(enable_fp_fusion=True, disable_ftz=False,
                                                       launch_pdl=False, native_matmul=False)
                or arithmetic.helper_sha256() != _HELPERS_SHA):
            raise ValueError('Block-FP8 embedding RMS boundary changed')
    else:
        arithmetic.validate(spec, boundary, selected)
    changed = arithmetic.transform(source, name, spec)
    if canonical_function_sha256(changed, name) != spec['function_sha256']:
        raise RuntimeError('Block-FP8 ordered RMS arithmetic changed')
    return changed


def verify_helpers(fn):
    if arithmetic.helper_sha256() != _HELPERS_SHA:
        raise RuntimeError('Qualified block-FP8 reduction helper source changed')
    for node in ast.parse(arithmetic.helper_source()).body:
        if isinstance(node, ast.FunctionDef):
            helper = fn.fn.__globals__.get(node.name)
            if helper is None or canonical_function_sha256(helper.src) != canonical_function_sha256(ast.unparse(node)):
                raise RuntimeError('Block-FP8 ordered RMS helper binding changed')


def prepare(fn, row, boundary):
    """All AsyncCompile/direct-PyCodeCache paths instantiate this same tuner.

    Keep its original numerical descriptor/config receipt and give Torch a new
    JIT function. Never mutate a JIT object already used by another constructor.
    """
    body = canonical_function_sha256(fn.src)
    selected = row['selected_fields']
    before = original_body(body, selected)
    spec = specification(before, selected)
    if spec is None:
        return fn
    if body != before:
        verify_helpers(fn)
        return fn
    source = rewrite(fn.src, fn.__name__, boundary, selected)
    tree = ast.parse(source)
    target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn.__name__)
    target.decorator_list = [ast.parse('triton.jit', mode='eval').body]
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    directory = Path(os.environ.get('TORCHINDUCTOR_CACHE_DIR', tempfile.gettempdir())) / 'mach_fp8_order'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(source.encode()).hexdigest() + '.py')
    if path.exists():
        if path.read_text() != source:
            raise RuntimeError('Block-FP8 ordered source cache changed')
    else:
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(source)
        os.replace(temporary, path)
    namespace = dict(fn.fn.__globals__)
    exec(compile(source, str(path), 'exec'), namespace)
    rebuilt = namespace[fn.__name__]
    if (rebuilt.arg_names != fn.arg_names or rebuilt.constexprs != fn.constexprs
            or canonical_function_sha256(rebuilt.src) != spec['function_sha256']):
        raise RuntimeError('Block-FP8 ordered RMS JIT ABI changed')
    verify_helpers(rebuilt)
    return rebuilt
