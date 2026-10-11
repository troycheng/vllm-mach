# SPDX-License-Identifier: Apache-2.0
"""Exact FULL16/FULL24 producer scheduling for the Qwen3.5-2B profile.

The package owns its finite source/config policy. No previous compiler cache,
observer, experiment directory or extra checkpoint is required at startup.
"""
from collections import Counter
from functools import wraps
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

from ..compile_choices import recipe_from_source

_POLICY_PATH = Path(__file__).with_name('data') / 'full_pdl.json'
_INSTALLED = False
_READY = False
_POLICY = None
_PENDING = {}
_CLONES = {}
_COMPILED = []
_COUNTS = Counter()
_SELECTED = {}
_DESCRIPTOR_LOCAL = threading.local()
_FULL_MODE = None
_CAPTURING = lambda: False


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _config(config):
    return {name: dict(getattr(config, name)) if name == 'kwargs' else getattr(config, name)
            for name in ('kwargs', 'num_warps', 'num_stages', 'num_ctas')}


def _config_key(config):
    return _sha(json.dumps(config, sort_keys=True, separators=(',', ':')))[:16]


def _kernel_key(spec, rows, config):
    return spec['source_sha256'], rows, _config_key(config)


def policy():
    global _POLICY
    if _POLICY is None:
        data = json.loads(_POLICY_PATH.read_text())
        if data['version'] != 1 or data['rows'] != [16, 24] or len(data['kernels']) != 7:
            raise RuntimeError('Unsupported 2B PDL policy')
        for item in data['kernels']:
            if (_sha(item['original_source']) != item['source_sha256']
                    or _sha(item['source']) != item['pdl_source_sha256']):
                raise RuntimeError('2B PDL source integrity mismatch')
            before = ast.parse(item['original_source'])
            after = ast.parse(item['source'])
            fn = next(n for n in after.body if isinstance(n, ast.FunctionDef))
            control = [n for n in fn.body if isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Call) and ast.unparse(n.value.func)
                       in ('tl.extra.cuda.gdc_wait', 'tl.extra.cuda.gdc_launch_dependents')]
            if len(control) != 2:
                raise RuntimeError('PDL requires one wait and one dependent launch')
            fn.body = [n for n in fn.body if n not in control]
            changed = 0
            for node in ast.walk(after):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value == 'launch_pdl':
                            if not isinstance(value, ast.Constant) or value.value is not True:
                                raise RuntimeError('PDL launch metadata changed')
                            value.value = False
                            changed += 1
            if changed != 1 or ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
                raise RuntimeError('PDL source changes arithmetic or metadata')
            recipe = recipe_from_source(item['original_source'], item['name'], {}, item['name'])
            if recipe['semantic_key'] != item['semantic_key'] or recipe['boundary'] != item['boundary']:
                raise RuntimeError('PDL semantic policy mismatch')
        _POLICY = data
    return _POLICY


def _target(name):
    return (name.startswith('triton_red_fused__to_copy_add_fused_add_rms_norm_')
            or name.startswith('triton_poi_fused_mul_silu_slice_'))


def _descriptor(name, source):
    if not _target(name):
        return None
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    names = [arg.arg for arg in fn.args.args]
    kind = 'swiglu' if 'silu_slice' in name else 'norm'
    width = None
    if kind == 'swiglu':
        widths = {n.right.value for n in ast.walk(fn) if isinstance(n, ast.BinOp)
                  and isinstance(n.op, ast.Mod) and isinstance(n.right, ast.Constant)
                  and isinstance(n.right.value, int) and n.right.value > 1}
        if len(widths) == 1:
            width = widths.pop()
    recipe = recipe_from_source(source, name, {}, name)
    matches = [s for s in policy()['kernels'] if s['semantic_key'] == recipe['semantic_key']]
    exact = next((s for s in matches if s['source_sha256'] == _sha(source)), None)
    return dict(arg_names=names, kind=kind, width=width,
                spec=exact or (matches[0] if matches else None), actual_source_sha256=_sha(source))


def _current_descriptor():
    stack = getattr(_DESCRIPTOR_LOCAL, 'stack', ())
    return stack[-1] if stack else None


def selected_full(rows):
    bound = _current_descriptor()
    return (_INSTALLED and rows in (16, 24) and _CAPTURING() and bound is not None
            and bound['mode'] == _FULL_MODE and bound['rows'] == rows and bound['warmup'] is False)


def native_selected(rows):
    selected = selected_full(rows)
    if selected:
        if not _READY:
            raise RuntimeError('2B PDL must be prepared before capture')
        _COUNTS[f'native_capture_m{rows}'] += 1
    return selected


def _bind_manager(manager):
    original = manager.capture

    @wraps(original)
    def capture(self, create_forward_fn, *args, **kwargs):
        @wraps(create_forward_fn)
        def factory(desc, *factory_args, **factory_kwargs):
            warmup = factory_kwargs.get('warmup', factory_args[0] if factory_args else None)
            if type(warmup) is not bool:
                raise RuntimeError('Graph capture must supply bool warmup')
            fn = create_forward_fn(desc, *factory_args, **factory_kwargs)
            bound = dict(mode=desc.cg_mode, rows=int(desc.num_tokens), warmup=warmup)

            @wraps(fn)
            def forward(*a, **kw):
                stack = getattr(_DESCRIPTOR_LOCAL, 'stack', None)
                if stack is None:
                    stack = _DESCRIPTOR_LOCAL.stack = []
                stack.append(bound)
                try:
                    return fn(*a, **kw)
                finally:
                    if stack.pop() is not bound:
                        raise RuntimeError('PDL capture descriptor nesting changed')
            return forward
        return original(self, factory, *args, **kwargs)
    manager.capture = capture


def _selected_config(kernel, spec):
    launchers = getattr(kernel, 'launchers', ())
    if len(launchers) != 1 or getattr(launchers[0], 'config', None) is None:
        raise RuntimeError('PDL requires the original warmup-selected launcher')
    config = _config(launchers[0].config)
    if config not in spec['allowed_configs']:
        raise RuntimeError('Unqualified 2B PDL launch configuration')
    return config


class _Proxy:
    def __init__(self, original, descriptor):
        self._original, self._descriptor = original, descriptor
        self._checked_rows = set()

    def __getattr__(self, name):
        return getattr(self._original, name)

    def run(self, *args, **kwargs):
        d = self._descriptor
        index = d['arg_names'].index('xnumel')
        rows = int(args[index] if index < len(args) else kwargs['xnumel'])
        if d['kind'] == 'swiglu':
            if not d['width'] or rows % d['width']:
                raise RuntimeError('PDL cannot resolve SwiGLU physical rows')
            rows //= d['width']
        spec = d['spec']
        if not spec:
            if selected_full(rows):
                raise RuntimeError('Unqualified producer source in FULL16/FULL24')
            return self._original.run(*args, **kwargs)
        if rows in (16, 24) and rows not in self._checked_rows:
            if _CAPTURING():
                raise RuntimeError('Original real warmup must precede PDL capture')
            result = self._original.run(*args, **kwargs)
            _selected_config(self._original, spec)
            self._checked_rows.add(rows)
            return result
        if selected_full(rows):
            config = _selected_config(self._original, spec)
            key = _kernel_key(spec, rows, config)
            if not _READY or key not in _CLONES:
                raise RuntimeError('PDL launcher was not precompiled')
            record = f'{spec["semantic_key"]}:m{rows}:config{_config_key(config)}'
            _SELECTED[record] = dict(semantic_key=spec['semantic_key'], rows=rows, config=config,
                                     source_sha256=spec['source_sha256'], actual_source_sha256=d['actual_source_sha256'])
            _COUNTS[f'producer_capture_m{rows}'] += 1
            return _CLONES[key].run(*args, **kwargs)
        return self._original.run(*args, **kwargs)


def _allocate(layout):
    import torch
    args = []
    for item in layout['argument_layout']:
        if not isinstance(item, dict):
            args.append(item)
        else:
            count = item['storage_offset'] + 1 + sum((n - 1) * s for n, s in zip(item['shape'], item['stride']))
            base = torch.zeros(count, dtype=torch.bfloat16, device='cuda')
            args.append(torch.as_strided(base, item['shape'], item['stride'], item['storage_offset']))
    return args


def prepare():
    """Precompile all qualified variants after CUDA setup, before model capture."""
    global _READY
    if _READY:
        return
    import torch
    import triton
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from .. import native_backend
    data = policy()
    if torch.__version__ != data['torch_version'] or triton.__version__ != data['triton_version']:
        raise RuntimeError('2B PDL compiler version differs from its qualified policy')
    if _CAPTURING() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('Prepare 2B PDL on SM120 before capture')
    native_backend.verify_runtime()
    if 'bool use_pdl=False' not in str(torch.ops.mxfp8_sm120.gemm.default._schema):
        raise RuntimeError('Install native MXFP8 with the use_pdl GEMM API')
    root = Path(os.environ['TORCHINDUCTOR_CACHE_DIR']) / 'mach_full_pdl'
    root.mkdir(parents=True, exist_ok=True)
    for spec in data['kernels']:
        path = root / (spec['pdl_source_sha256'] + '.py')
        path.write_text(spec['source'])
        for layout in spec['layouts']:
            for config in spec['allowed_configs']:
                key = _kernel_key(spec, layout['rows'], config)
                module_name = '_mach_pdl_' + _sha(str(key))
                module_spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[module_name] = module
                module_spec.loader.exec_module(module)
                kernel = getattr(module, spec['name'])
                if not isinstance(kernel, CachingAutotuner) or kernel.launchers or kernel.compile_results:
                    raise RuntimeError('PDL requires a fresh CachingAutotuner')
                kernel.configs = [triton.Config(config['kwargs'], num_warps=config['num_warps'],
                                               num_stages=config['num_stages'], num_ctas=config['num_ctas'])]
                for name in ('dynamic_scale_rblock', 'coordinate_descent_tuning', 'dynamic_disable_pipelining'):
                    kernel.inductor_meta[name] = False
                kernel.save_cache_hook = None
                meta = copy.deepcopy(kernel.triton_meta)
                kernel.precompile()
                if (len(kernel.launchers) != 1 or len(kernel.compile_results) != 1
                        or _config(kernel.launchers[0].config) != config
                        or kernel.triton_meta != meta or not meta.get('launch_pdl')):
                    raise RuntimeError('PDL compiler changed its qualified configuration')
                binary = kernel.compile_results[0].kernel
                if hasattr(binary, 'asm'):
                    ptx = binary.asm['ptx']
                else:
                    from torch._inductor.runtime.runtime_utils import triton_cache_dir, triton_hash_to_path_key
                    cache = Path(triton_cache_dir(0)) / triton_hash_to_path_key(binary.hash)
                    ptx = (cache / f'{binary.name}.ptx').read_text()
                if not all(op in ptx for op in ('griddepcontrol.wait', 'griddepcontrol.launch_dependents')):
                    raise RuntimeError('PDL compiled kernel lacks dependency instructions')
                _COMPILED.append(dict(source_sha256=spec['source_sha256'], rows=layout['rows'],
                                      config=config, ptx_sha256=_sha(ptx), dependency_instructions=True))
                args = _allocate(layout)
                kernel.run(*args, stream=torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()  # Startup scratch lifetime only.
                _CLONES[key] = kernel
    if len(_CLONES) != 20:
        raise RuntimeError('Incomplete PDL precompilation')
    _READY = True


def install():
    global _INSTALLED, _FULL_MODE, _CAPTURING
    if _INSTALLED:
        return
    import torch
    from torch._inductor.async_compile import AsyncCompile
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from .. import native_backend
    policy()
    _FULL_MODE = CUDAGraphMode.FULL
    _CAPTURING = torch.cuda.is_current_stream_capturing
    original_triton, original_wait = AsyncCompile.triton, AsyncCompile.wait

    @wraps(original_triton)
    def triton(self, name, source, *args, **kwargs):
        descriptor = _descriptor(name, source)
        result = original_triton(self, name, source, *args, **kwargs)
        if descriptor is not None:
            _PENDING[id(result)] = (result, descriptor)
        return result

    @wraps(original_wait)
    def wait(self, scope, *args, **kwargs):
        before = {name: _PENDING[id(value)][1] for name, value in list(scope.items()) if id(value) in _PENDING}
        result = original_wait(self, scope, *args, **kwargs)
        for name, descriptor in before.items():
            if not callable(getattr(scope[name], 'run', None)):
                raise RuntimeError('Compiled producer has no launcher')
            if not isinstance(scope[name], _Proxy):
                scope[name] = _Proxy(scope[name], descriptor)
        return result

    _bind_manager(CudaGraphManager)
    native_backend.set_pdl_selector(native_selected)
    AsyncCompile.triton, AsyncCompile.wait = triton, wait
    _INSTALLED = True


def verify_capture(required_rows):
    expected = set(required_rows) & {16, 24}
    if not _INSTALLED or not _READY:
        raise RuntimeError('2B PDL startup incomplete')
    semantics = {s['semantic_key'] for s in policy()['kernels']}
    for rows in expected:
        observed = {s['semantic_key'] for s in _SELECTED.values() if s['rows'] == rows}
        if (observed != semantics or not _COUNTS[f'native_capture_m{rows}']
                or not _COUNTS[f'producer_capture_m{rows}']):
            raise RuntimeError(f'Incomplete 2B FULL{rows} PDL capture')
    return receipt()


def receipt():
    return dict(installed=_INSTALLED, ready=_READY, rows=[16, 24],
                policy_sha256=hashlib.sha256(_POLICY_PATH.read_bytes()).hexdigest(),
                precompiled_variants=len(_CLONES), compiled=list(_COMPILED), selected_configs=list(_SELECTED.values()),
                counts=dict(_COUNTS), selection='FULL16/FULL24 capture only',
                counter_scope='Python capture calls; not graph replay counts')
