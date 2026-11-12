# SPDX-License-Identifier: Apache-2.0
"""Qualified stateless Block-FP8 islands on the exact2048 PIECEWISE route."""
import ast
from dataclasses import replace
from functools import wraps
import hashlib
import inspect as python_inspect
import json
import os
from pathlib import Path
import threading

_DATA = Path(__file__).parent / "data/prefill_graph.json"
_BASE = (1, 2, 4, *range(8, 257, 8))
_STATES = {}
_TLS = threading.local()
_HOOKS = []


def ast_sha(node):
    def normalize(value):
        if isinstance(value, ast.AST):
            return {"node": type(value).__name__, "fields": {
                k: normalize(v) for k, v in ast.iter_fields(value)
                if not (k == "type_params" and v == [])}}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    return hashlib.sha256(json.dumps(normalize(node), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def aot_identity(source):
    tree = ast.parse(source)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "call")
    kernels = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == "async_compile.triton"):
            kernel = ast.parse(ast.literal_eval(n.value.args[1]))
            fn = next(x for x in kernel.body if isinstance(x, ast.FunctionDef))
            kernels[n.targets[0].id] = ast_sha(fn)
    return ast_sha(call), kernels


def pure_fx(graph):
    """Finite observed call targets; AOT identity additionally locks all math."""
    if graph is None:
        return False
    opaque = {"vllm_mach::block_fp8_linear", "vllm_mach_fp8::silu_and_mul_per_block_quant",
              "_C::cutlass_scaled_mm", "_C::per_token_group_fp8_quant",
              "vllm_ir::rms_norm", "vllm_ir::fused_add_rms_norm"}
    methods = {"float", "to", "view", "reshape", "flatten", "transpose", "permute",
               "contiguous", "size", "stride", "numel", "mean", "pow", "silu",
               "unsqueeze", "squeeze", "split", "chunk"}
    pure = {"add", "mul", "sub", "truediv", "floordiv", "getitem", "getattr", "sqrt",
            "rsqrt", "mean", "pow", "cat", "split", "chunk", "stack", "empty",
            "empty_like", "zeros", "zeros_like", "ones", "ones_like", "arange",
            "reshape", "permute", "transpose", "silu", "linear", "sym_int",
            "sym_float", "sym_max", "sym_min"}
    for n in graph.graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        if n.op == "call_method":
            if str(n.target) not in methods:
                return False
        elif n.op == "call_function":
            target = n.target
            schema = getattr(target, "_schema", None)
            if schema is None and getattr(target, "_schemas", None) is not None:
                schema = getattr(getattr(target, "default", None), "_schema", None)
            if schema is not None:
                name = schema.name
                if not (name.startswith(("aten::", "prims::")) or name in opaque):
                    return False
                if any(t in name for t in ("random", "rand", "dropout", "bernoulli", "normal", "seed", "uniform")):
                    return False
            else:
                name = getattr(target, "__name__", str(target))
                module = getattr(target, "__module__", "")
                observed = ((module == "torch.nn.functional" and name == "silu")
                            or (module == "torch" and name in ("Size", "sigmoid")))
                if observed and (n.kwargs or len(n.args) != 1):
                    return False
                if not (observed or (module in ("operator", "_operator", "builtins", "math", "torch", "torch._C._nn") and name in pure)
                        or str(target) in ("triton_kernel_wrapper_mutation", "triton_kernel_wrapper_functional")):
                    return False
        else:
            # No module/attribute lookup can hide model-state reads or writes.
            return False
    return True


def eligible_config(cfg, features):
    if tuple(features) != ("n64", "ordered", "silu", "fa2") or os.environ.get("VLLM_MACH_FP8_COMPILE_MODE", "production") != "production":
        return False
    try:
        from types import SimpleNamespace
        from .worker import validate_worker_config
        validate_worker_config(SimpleNamespace(vllm_config=cfg))
        c = cfg.compilation_config
        expected = json.loads(_DATA.read_text())["compilation"]
        if c.mode.name != expected["mode"] or c.backend != expected["backend"]:
            return False
        for key in ("custom_ops", "splitting_ops", "inductor_compile_config"):
            if getattr(c, key) != expected[key]:
                return False
        if any(getattr(c.pass_config, k) != v for k, v in expected["pass_config"].items()):
            return False
        return (cfg.scheduler_config.max_num_seqs == 128
                and cfg.scheduler_config.max_num_batched_tokens == 2048
                and cfg.cache_config.kv_cache_memory_bytes == 19 * 2**30
                and cfg.model_config.max_model_len == 8192
                and c.cudagraph_mode.name == "FULL_AND_PIECEWISE"
                and tuple(c.cudagraph_capture_sizes) == _BASE
                and c.max_cudagraph_capture_size == 256
                and c.compile_sizes == [] and c.compile_ranges_endpoints == [2048]
                and not c.use_inductor_graph_partition)
    except (AttributeError, RuntimeError, TypeError):
        return False


def exact_descriptor(desc, tokens, requests):
    from vllm.config.compilation import CUDAGraphMode
    if tokens == 2048:
        if desc.cg_mode != CUDAGraphMode.PIECEWISE or desc.num_tokens != 2048:
            raise RuntimeError("Qualified Block2048 requires its PIECEWISE descriptor")
    elif desc.num_tokens == 2048:
        desc = replace(desc, cg_mode=CUDAGraphMode.NONE, num_tokens=tokens,
                       num_reqs=requests, uniform_token_count=None, max_query_len=None)
    return desc



def runtime_descriptor(state, desc, tokens, requests):
    desc = exact_descriptor(desc, tokens, requests)
    if state.ready and state.fallback and tokens == 2048:
        from vllm.config.compilation import CUDAGraphMode
        desc = replace(desc, cg_mode=CUDAGraphMode.NONE, num_tokens=tokens,
                       num_reqs=requests, uniform_token_count=None, max_query_len=None)
    return desc

def storage(tensor):
    return str(tensor.device), tensor.untyped_storage().data_ptr()


class State:
    def __init__(self, cfg):
        self.cfg = cfg
        self.manifest = json.loads(_DATA.read_text())
        self.islands = {}
        self.artifacts = {}
        self.wrappers = {}
        self.readonly = {}
        self.forbidden = set()
        self.bank = None
        self.ready = False
        self.fallback = None


def configure(worker, features):
    cfg = worker.vllm_config
    if not eligible_config(cfg, features):
        worker._mach_fp8_prefill_graph_disabled = "unqualified_config"
        return False
    import importlib.metadata as md
    manifest = json.loads(_DATA.read_text())
    for relative, expected in manifest["upstream_sources"].items():
        try:
            actual = Path(md.distribution("vllm").locate_file(relative))
            matches = hashlib.sha256(actual.read_bytes()).hexdigest() == expected
        except (OSError, md.PackageNotFoundError):
            matches = False
        if not matches:
            worker._mach_fp8_prefill_graph_disabled = "upstream_source_mismatch:" + relative
            return False
    state = State(cfg)
    _STATES[id(cfg)] = state
    worker._mach_fp8_prefill_graph = state
    cfg.compilation_config.cudagraph_capture_sizes = [*_BASE, 2048]
    cfg.compilation_config.max_cudagraph_capture_size = 2048
    install_hooks()
    return True


def bind_model(worker):
    state = getattr(worker, "_mach_fp8_prefill_graph", None)
    if state is None:
        return
    import torch
    runner = worker.model_runner
    state.readonly = {storage(t): name for name, t in runner.model.named_parameters()}
    state.forbidden.clear()
    for name, tensor in runner.model.named_buffers():
        if name.endswith(("_mach_block_fp8_scale64", ".weight_scale", ".weight_scale_inv")):
            state.readonly[storage(tensor)] = name
        else:
            state.forbidden.add(storage(tensor))
    def collect(value, depth=0):
        if isinstance(value, torch.Tensor):
            state.forbidden.add(storage(value))
        elif depth <= 3 and isinstance(value, (list, tuple, dict)):
            for child in (value.values() if isinstance(value, dict) else value):
                collect(child, depth + 1)
    collect(getattr(runner, "kv_caches", []))
    for owner in (getattr(runner, "req_states", None), getattr(runner, "model_state", None)):
        if owner is not None:
            for value in vars(owner).values():
                collect(value)
                if hasattr(value, "__dict__"):
                    collect(vars(value).get("gpu"))


def before_compile(worker):
    state = getattr(worker, "_mach_fp8_prefill_graph", None)
    if state is not None:
        bind_model(worker)
        state.main_stream = worker.model_runner.main_stream


def after_compile(worker):
    state = getattr(worker, "_mach_fp8_prefill_graph", None)
    if state is None:
        return
    if state.bank is not None:
        state.bank.bind_runtime(state.main_stream)
    if not any(row["safe"] for row in state.wrappers.values()):
        state.fallback = "no_source_qualified_islands"
    state.ready = True
    if all(s.ready for s in _STATES.values()):
        for obj, name, original, replacement in _HOOKS:
            current = getattr(obj, name)
            if isinstance(original, classmethod):
                current = current.__func__
            if current is replacement:
                setattr(obj, name, original)
            else:
                state.fallback = "startup_hook_changed"
        _HOOKS.clear()


def inspect_worker(worker):
    state = getattr(worker, "_mach_fp8_prefill_graph", None)
    if state is None:
        return {"enabled": False, "fallback": getattr(worker, "_mach_fp8_prefill_graph_disabled", None)}
    return {"enabled": state.fallback is None, "ready": state.ready, "fallback": state.fallback,
            "qualified_islands": sum(r["safe"] for r in state.wrappers.values()),
            "bank_bytes": 0 if state.bank is None else state.bank.bytes,
            "manifest_sha256": hashlib.sha256(_DATA.read_bytes()).hexdigest()}


def install_hooks():
    from torch._inductor.codecache import PyCodeCache
    from vllm.compilation.piecewise_backend import PiecewiseBackend
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if _HOOKS:
        return
    original_load = PyCodeCache.load_by_key_path.__func__
    @wraps(original_load)
    def load(cls, *args, **kwargs):
        module = original_load(cls, *args, **kwargs)
        row = getattr(_TLS, "active", None)
        if row is not None:
            path = kwargs.get("path", args[1] if len(args) > 1 else module.__file__)
            source = Path(path).read_text()
            if "class Runner" in source and "def call(" in source:
                try:
                    key, kernels = aot_identity(source)
                    if kernels != row["state"].manifest["families"].get(key):
                        row["invalid"] = True
                    else:
                        row["aot"].append(key)
                except (ValueError, SyntaxError, StopIteration, TypeError):
                    row["invalid"] = True
        return module
    PyCodeCache.load_by_key_path = classmethod(load)
    _HOOKS.append((PyCodeCache, "load_by_key_path", classmethod(original_load), load))
    original_init = PiecewiseBackend.__init__
    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        bound = python_inspect.signature(original_init).bind(self, *args, **kwargs)
        bound.apply_defaults()
        state = _STATES.get(id(bound.arguments["vllm_config"]))
        if state is None:
            return original_init(self, *args, **kwargs)
        row = {"state": state, "fx": pure_fx(bound.arguments["graph"]),
               "aot": [], "invalid": False, "safe": False}
        previous = getattr(_TLS, "active", None)
        _TLS.active = row
        try:
            original_init(self, *args, **kwargs)
        finally:
            _TLS.active = previous
        entries = list(self.range_entries.values())
        if not row["aot"]:
            inherited = [state.artifacts.get(id(e.runnable)) for e in entries]
            if inherited and all(x is not None for x in inherited):
                row["aot"] = [key for group in inherited for key in group]
        row["safe"] = row["fx"] and bool(row["aot"]) and not row["invalid"]
        if row["safe"]:
            for entry in entries:
                state.artifacts[id(entry.runnable)] = row["aot"]
        state.islands[id(self)] = row
    PiecewiseBackend.__init__ = initialize
    _HOOKS.append((PiecewiseBackend, "__init__", original_init, initialize))
    # These runtime wrappers stay installed, but only this config's qualified
    # instances are admitted. Other profiles and all previous descriptors use
    # the original entry without a new CUDA operation.
    if getattr(CUDAGraphWrapper, "_mach_block_pw2048", False):
        return
    dispatch = CudaGraphManager.dispatch
    @wraps(dispatch)
    def route(self, *args, **kwargs):
        desc = dispatch(self, *args, **kwargs)
        state = _STATES.get(id(self.vllm_config))
        if state is None or not self._graphs_captured:
            return desc
        tokens = kwargs.get("num_tokens", args[1] if len(args) > 1 else None)
        requests = kwargs.get("num_reqs", args[0] if args else None)
        return runtime_descriptor(state, desc, tokens, requests)
    CudaGraphManager.dispatch = route
    original_call = CUDAGraphWrapper.__call__
    @wraps(original_call)
    def call(self, *args, **kwargs):
        state = _STATES.get(id(self.vllm_config))
        if state is None or not is_forward_context_available():
            return original_call(self, *args, **kwargs)
        ctx = get_forward_context()
        desc = ctx.batch_descriptor
        if desc is None or desc.num_tokens != 2048 or ctx.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
            return original_call(self, *args, **kwargs)
        if kwargs:
            return self.runnable(*args, **kwargs)
        row = state.wrappers.get(id(self))
        if row is None:
            if state.ready:
                return self.runnable(*args, **kwargs)  # never capture a first user request
            import torch
            backend = state.islands.get(id(self.runnable), {})
            safe = backend.get("safe", False) and not kwargs
            tensors = [t for t in args if isinstance(t, torch.Tensor)]
            safe = safe and not any(storage(t) in state.forbidden for t in tensors)
            safe = safe and not any(storage(t) not in state.readonly and t.untyped_storage().nbytes() > 512 * 2**20 for t in tensors)
            row = {"safe": bool(safe)}
            state.wrappers[id(self)] = row
        if not row["safe"]:
            return self.runnable(*args, **kwargs)
        if state.bank is None:
            from ._prefill_inputs import ActivationBank
            import torch
            state.bank = ActivationBank(next(t.device for t in args if isinstance(t, torch.Tensor)))
        prepared, _ = state.bank.prepare(args, state.readonly, row.get("signature"))
        if prepared is None:
            if not state.ready:
                row["safe"] = False
            return self.runnable(*args, **kwargs)
        fixed, signature = prepared
        if not state.ready:
            row["signature"] = signature
        # Original wrapper capture/replay/output weakref/lifetime semantics.
        return original_call(self, *fixed, **kwargs)
    CUDAGraphWrapper.__call__ = call
    CUDAGraphWrapper._mach_block_pw2048 = True
