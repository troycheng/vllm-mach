# SPDX-License-Identifier: Apache-2.0
"""Finite 2B compiler ordering rules; independent of teacher/quality tooling."""
import ast
import functools
import hashlib
import inspect as pyinspect
import json
import os
from pathlib import Path
import threading
import weakref

from ..compile_choices import canonical_function_sha256, recipe_from_source
from ..ordered_reduction import transform_2b

_PATH = Path(__file__).with_name('data') / 'ordered_reduction.json'
_HELPERS = Path(__file__).parents[1] / '_ordered_reduction_2b_kernel.py'
_INSTALLED = False
_OBJECTS = []
_TEMPORARY = []
_LOCAL = threading.local()
_REWRITES = 0


def policy():
    data = json.loads(_PATH.read_text())
    if data['format_version'] != 1 or hashlib.sha256(_HELPERS.read_bytes()).hexdigest() != data['helpers_sha256']:
        raise RuntimeError('2B ordered reduction policy/helper differs')
    return data


def _boundaries(row):
    return [{**row['boundary'], 'size_hints': hints} for hints in row['size_hints']]


def _geometry(boundary):
    return {k: v for k, v in boundary.items() if k not in ('function_sha256', 'compiler_flags')}


def rewrite(source, name):
    data = policy()
    recipe = recipe_from_source(source, name, {}, name)
    boundary = recipe['boundary']
    body = boundary['function_sha256']
    original = data['targets'].get(body)
    if original is None:
        original = next((v for v in data['targets'].values() if v['after_sha256'] == body), None)
        if original is not None:
            if boundary not in [{**b, 'function_sha256': body} for b in _boundaries(original)]:
                raise RuntimeError('Changed transformed 2B norm boundary')
            functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
            for node in ast.parse(_HELPERS.read_text()).body:
                if isinstance(node, ast.FunctionDef) and (node.name not in functions or
                        canonical_function_sha256(ast.unparse(functions[node.name])) !=
                        canonical_function_sha256(ast.unparse(node))):
                    raise RuntimeError('Changed transformed 2B norm helper')
            return source
        if (body not in data['allowed_unmodified'] and
                (_geometry(boundary) in [_geometry(b) for v in data['targets'].values() for b in _boundaries(v)]
                 or name.startswith(('triton_red_fused__to_copy_add_fused_add_rms_norm_',
                                     'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_silu_')))):
            raise RuntimeError('Unqualified 2B norm source/geometry')
        return source
    if boundary not in _boundaries(original):
        raise RuntimeError('Unqualified 2B ordered norm descriptor')
    result = transform_2b(source, name, data['targets'])
    if recipe_from_source(result, name, {}, name)['boundary'] != {
            **boundary, 'function_sha256': original['after_sha256']}:
        raise RuntimeError('2B norm transform changed its numerical boundary')
    return result


def _config(value):
    return dict(kwargs=dict(value.kwargs), num_warps=value.num_warps,
                num_stages=value.num_stages, num_ctas=getattr(value, 'num_ctas', 1))


def _validate_globals(fn):
    for node in ast.parse(_HELPERS.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            helper = fn.fn.__globals__.get(node.name)
            if helper is None or canonical_function_sha256(helper.src) != canonical_function_sha256(ast.unparse(node)):
                raise RuntimeError('2B ordered norm JIT helper differs')


def _save_diagnostic(reason):
    """Preserve actual post-precompile evidence before a cold-load guard fails."""
    directory = os.environ.get('VLLM_MACH_MXFP8_RUN_DIR')
    if directory is None:
        return
    rows = []
    for body, ref in _OBJECTS:
        obj = ref()
        if obj is None:
            continue
        rows.append(dict(body_sha256=body, source=obj.fn.src,
                         size_hints=obj.size_hints,
                         compiler_flags={k: obj.triton_meta.get(k, False) for k in
                                         ('enable_fp_fusion', 'disable_ftz', 'launch_pdl', 'native_matmul')},
                         constructor_configs=obj._mach_2b_norm_constructor_configs,
                         current_configs=None if obj.configs is None else [_config(x) for x in obj.configs],
                         launchers=[_config(x.config) for x in obj.launchers],
                         compiled=[dict(config=_config(x.config), kernel_hash=getattr(x.kernel, 'hash', None))
                                   for x in obj.compile_results], filename=str(obj.filename)))
    path = Path(directory) / ('ordered-norm-diagnostic-' + str(os.getpid()) + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(reason=reason, actual=rows,
                                    helper_source=_HELPERS.read_text(),
                                    temporary_benchmark_constructors=len(_TEMPORARY)), indent=2) + '\n')


def install():
    global _INSTALLED, _REWRITES
    if os.environ.get('VLLM_MACH_PROFILE') != 'qwen35-2b-mxfp8-champion-v1':
        raise RuntimeError('Explicit 2B profile required for ordered reductions')
    if _INSTALLED:
        return False
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from triton import Config
    data = policy()
    after = {v['after_sha256']: v for v in data['targets'].values()}
    original_compile = AsyncCompile.triton
    original_load = PyCodeCache.load
    original_init = CachingAutotuner.__init__
    benchmark = TritonScheduling.benchmark_fused_nodes
    benchmark_path = Path(pyinspect.getsourcefile(benchmark))
    if hashlib.sha256(benchmark_path.read_bytes()).hexdigest() != data['torch_benchmark_codegen_sha256']:
        raise RuntimeError('2B direct benchmark compiler source differs')

    @functools.wraps(original_compile)
    def compile_triton(self, name, source, *args, **kwargs):
        global _REWRITES
        if name.startswith('triton_'):
            changed = rewrite(source, name)
            _REWRITES += changed != source
            source = changed
        return original_compile(self, name, source, *args, **kwargs)

    def benchmark_context(source):
        frame = pyinspect.currentframe().f_back
        try:
            while frame is not None:
                if frame.f_code is benchmark.__code__:
                    if frame.f_locals.get('src_code') != source:
                        raise RuntimeError('Direct benchmark source differs from actual compiler frame')
                    return True
                frame = frame.f_back
        finally:
            del frame
        return False

    @classmethod
    @functools.wraps(original_load.__func__)
    def load(cls, source, *args, **kwargs):
        global _REWRITES
        tree = ast.parse(source)
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        targets = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('triton_'):
                try:
                    body = canonical_function_sha256(source, node.name)
                except (SyntaxError, ValueError, StopIteration):
                    continue
                if body in data['targets'] or body in after:
                    targets.append(node.name)
                else:
                    # Unknown placeholder names must not bypass source/FP-flag
                    # guards just because they do not have a production alias.
                    rewrite(source, node.name)
        if len(targets) > 1:
            raise RuntimeError('Multiple unqualified direct norm constructions')
        temporary = False
        if targets:
            structural = {'get_args', 'call', 'benchmark_all_configs'} <= names
            temporary = benchmark_context(source)
            if structural != temporary or (temporary and targets != ['triton_']):
                raise RuntimeError('Unclassified direct 2B norm constructor')
            if temporary and recipe_from_source(source, 'triton_', {}, 'triton_')['kernel_site'] != 'Placeholder.DESCRIPTIVE_NAME':
                raise RuntimeError('Unexpected direct benchmark site')
            changed = rewrite(source, targets[0]);_REWRITES += changed != source;source = changed
        prior = getattr(_LOCAL, 'temporary', False)
        _LOCAL.temporary = temporary
        try:
            return original_load.__func__(cls, source, *args, **kwargs)
        finally:
            _LOCAL.temporary = prior

    signature = pyinspect.signature(original_init)

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs);bound.apply_defaults()
        fn = bound.arguments['fn'];body = canonical_function_sha256(fn.src)
        if body in data['targets']:
            raise RuntimeError('2B norm bypassed finite source transform')
        selected = None
        if body in after:
            expected = after[body]
            metadata = bound.arguments['triton_meta']
            if (metadata['signature'] != expected['boundary']['signature']
                    or {k: metadata[k] for k in expected['boundary']['compiler_flags']} != expected['boundary']['compiler_flags']
                    or bound.arguments['size_hints'] not in [b['size_hints'] for b in _boundaries(expected)]):
                raise RuntimeError('Actual 2B norm compiler metadata differs')
            if not getattr(_LOCAL, 'temporary', False):
                # Match the already-observed production/M32/M64 winner, as the
                # 4B finite selector does. Fusion benchmark candidates remain
                # temporary and never count as selected model coverage.
                selected = Config(**expected['selected'])
                bound.arguments['configs'] = [selected]
                bound.arguments['inductor_meta'] = {
                    **(bound.arguments.get('inductor_meta') or {}),
                    'dynamic_scale_rblock': False, 'coordinate_descent_tuning': False,
                    'incremental_autotune': False,
                }
        result = original_init(*bound.args, **bound.kwargs)
        if selected is not None:
            self.configs = [selected]
        if body in after:
            _validate_globals(fn)
            self._mach_2b_norm_constructor_configs = [_config(c) for c in self.configs]
            target = _TEMPORARY if getattr(_LOCAL, 'temporary', False) else _OBJECTS
            target.append((body, weakref.ref(self)))
        return result

    AsyncCompile.triton = compile_triton
    PyCodeCache.load = load
    CachingAutotuner.__init__ = initialize
    _INSTALLED = True
    return True


def receipt(require_complete=False):
    data = policy();rows = []
    for body, ref in _OBJECTS:
        obj = ref()
        if obj is None:
            continue
        if canonical_function_sha256(obj.fn.src) != body:
            raise RuntimeError('Actual 2B norm JIT changed after construction')
        _validate_globals(obj.fn)
        selected = [_config(x.config) for x in obj.launchers]
        expected = next(v['selected'] for v in data['targets'].values() if v['after_sha256'] == body)
        if selected and selected != [expected]:
            _save_diagnostic('Unqualified actual 2B norm selected configuration')
            raise RuntimeError('Unqualified actual 2B norm selected configuration')
        rows.append(dict(body_sha256=body, constructor_configs=obj._mach_2b_norm_constructor_configs,
                         current_configs=None if obj.configs is None else [_config(x) for x in obj.configs],
                         actual_launcher_configs=selected))
    if require_complete and {r['body_sha256'] for r in rows if r['actual_launcher_configs']} != {
            v['after_sha256'] for v in data['targets'].values()}:
        _save_diagnostic('Missing actual 2B ordered norm coverage')
        raise RuntimeError('Missing actual 2B ordered norm coverage')
    return dict(installed=_INSTALLED, policy_sha256=hashlib.sha256(_PATH.read_bytes()).hexdigest(),
                helper_sha256=data['helpers_sha256'], rewrites=_REWRITES,
                hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                actual=rows, temporary_benchmark_constructors=len(_TEMPORARY),
                scope='Ordinary warmup-selected launchers; not graph replay counts',
                pdl_clones='Unchanged separately source/config-bound PDL policy')
