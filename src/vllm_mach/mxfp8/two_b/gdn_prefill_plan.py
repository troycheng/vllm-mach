# SPDX-License-Identifier: Apache-2.0
"""Reuse FlashInfer's four compiled SM120 CP launches across matching 2B layers."""
from collections import Counter
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import RLock, get_ident
from types import FunctionType

SOURCE = "76f00777b51a7676bda53cefb4bdd543ace5c40d6cd85587b10d09fd6a9576d9"
UTILS = "8e900286e9175579e34f0a63c62ec2cedf420e5de6953f411964cc4424cf2fd2"
PUBLIC = "56eea898c26ead1bb5e2b00ea0f75eb09448c88bf092dbd2414f17bdebd11c42"
STAGES = ("t_precompute", "mn_precompute", "fixup", "prefill")
BUFFERS = ("t", "local_transfer", "local_state", "fixed_state", "prefill_tensormaps")
# Positional ABI of the four compiled callables in SOURCE. Each map identifies
# caller-owned tensors by CP input position (initial_state is slot 8).
BINDINGS = ({0: 3, 1: 6, 3: 7}, {0: 3, 1: 4, 3: 5, 6: 7}, {2: 8, 7: 7},
            {0: 2, 1: 3, 2: 4, 4: 0, 5: 5, 6: 1, 8: 8, 13: 7})
WORKSPACE = ({2: 0}, {2: 0, 4: 1, 5: 2}, {0: 1, 1: 2, 4: 3}, {3: 0, 7: 3, 12: 4})
ARITY = (10, 15, 13, 26)
KEYWORDS = {"initial_state", "state_indices", "state_checkpoints", "checkpoint_cu_starts",
            "checkpoint_every_n_tokens", "max_seqlen", "cp_chunk_len"}
_PROBE_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError, OSError, MemoryError)
_adapter = None
_reason = "not installed"


def _clone(fn, namespace):
    copy = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    copy.__kwdefaults__ = fn.__kwdefaults__
    return copy


def _recordable(stock):
    # Only these private function copies use the recorder. Every other caller
    # retains the original FlashInfer globals, even while a shape is priming.
    namespace = dict(stock.__globals__)
    calls = []
    def observe(getter):
        def lookup(*args, **kwargs):
            kernel = getter(*args, **kwargs)
            if kernel is None:
                return None
            def run(*values, **kwargs):
                try:
                    calls.append((kernel, values if not kwargs else ()))
                except MemoryError:
                    pass
                return kernel(*values, **kwargs)
            return run
        return lookup
    for name in ("get_cached_compile", "cached_compile"):
        namespace[name] = observe(namespace[name])
    for stage in STAGES:
        name = f"cp_delta_rule_{stage}_dsl_sm120"
        namespace[name] = _clone(namespace[name], namespace)
    return _clone(stock, namespace), calls


def _make_plan(calls, sources, owners, torch, stream):
    if len(calls) != 4 or any(buf is None for buf in owners):
        return None
    launches = []
    for (kernel, values), binding, work, arity in zip(calls, BINDINGS, WORKSPACE, ARITY, strict=True):
        if len(values) != arity or int(values[-1]) != stream:
            return None
        constants = list(values)
        views = []
        for i, value in enumerate(values):
            if i in binding:
                source_id = binding[i]
                src = sources[source_id]
                if (not isinstance(value, torch.Tensor) or value.data_ptr() != src.data_ptr()
                        or value.dtype != src.dtype):
                    return None
                view = None if value.shape == src.shape and value.stride() == src.stride() else (
                    tuple(value.shape), value.stride())
                views.append((i, source_id, view))
                constants[i] = None
            elif i in work:
                if not isinstance(value, torch.Tensor) or value.data_ptr() != owners[work[i]].data_ptr():
                    return None
            elif i != arity - 1 and value is not None and type(value) not in (int, float, bool):
                return None
            elif isinstance(value, torch.Tensor):
                return None
        launches.append((kernel, constants, views))
    return launches


def _bind(plan, sources):
    args_by_stage = []
    for kernel, constants, views in plan:
        args = constants.copy()
        for index, source_id, view in views:
            src = sources[source_id]
            args[index] = src if view is None else src.as_strided(*view)
        args_by_stage.append((kernel, args))
    return args_by_stage


class _Adapter:
    def __init__(self, stock, buffers, device, torch):
        self.stock = stock
        self.trace, self.calls = _recordable(stock)
        self.torch, self.buffers, self.device = torch, buffers, device
        self.names = tuple(("gdn_cp_sm120_" + name, device) for name in BUFFERS)
        self.owner = self.stream = self.key = self.plan = self.owners = None
        self.lock = RLock()
        self.reason = None
        self.counts = Counter()
        self.peak_bytes = 0

    def _stop(self, reason):
        self.reason = reason
        self.key = self.plan = self.owners = None

    def _source_key(self, args, kw):
        torch = self.torch
        if len(args) != 9 or kw.keys() - KEYWORDS:
            return None
        if (kw.get("state_indices") is not None or kw.get("state_checkpoints") is not None
                or kw.get("checkpoint_cu_starts") is not None or kw.get("checkpoint_every_n_tokens", 0)
                or kw.get("cp_chunk_len") is not None):
            return None
        sources = args[:8] + (kw.get("initial_state"),)
        o, state, q, k, v, alpha, beta, cu, initial = sources
        if not all(isinstance(t, torch.Tensor) and t.device == self.device for t in sources):
            return None
        # The qualified C16 turnover has one or two prefill sequences. The
        # public FI policy already selected CP and created its FP32 final state.
        if (q.ndim != 3 or q.dtype != torch.bfloat16 or tuple(q.shape[1:]) != (16, 128)
                or not 0 < q.shape[0] <= 2048 or cu.ndim != 1 or cu.numel() not in (2, 3)
                or state.dtype != torch.float32 or initial.dtype != torch.float32
                or type(args[8]) is not float or type(kw.get("max_seqlen")) is not int):
            return None
        key = (args[8], kw["max_seqlen"],
               tuple((t.dtype, t.shape, t.stride(), t.data_ptr() % 16) for t in sources))
        return key, sources

    def dispatch(self, *args, **kw):
        if self.reason:
            return self.stock(*args, **kw)
        with self.lock:
            if self.reason:
                return self.stock(*args, **kw)
            try:
                stream = self.torch.cuda.current_stream(self.device)
                capturing = self.torch.cuda.is_current_stream_capturing()
                if self.owner is not None and (stream != self.stream or get_ident() != self.owner):
                    self._stop("execution context changed")
                    if stream != self.stream and not capturing:
                        stream.wait_stream(self.stream)
                if self.reason or capturing:
                    info = None
                else:
                    info = self._source_key(args, kw)
            except _PROBE_ERRORS:
                self._stop("unsupported call metadata")
                info = None
            if info is None:
                return self.stock(*args, **kw)
            key, sources = info
            if self.owner is None:
                self.owner, self.stream = get_ident(), stream
            # Strong references keep the ABI views live. Match FI's current
            # allocations by identity as a different CP call can grow a buffer.
            if self.plan is not None and key == self.key and all(
                    self.buffers.get(name) is owner for name, owner in zip(self.names, self.owners)):
                try:
                    bound = _bind(self.plan, sources)
                except _PROBE_ERRORS:
                    self._stop("unsupported tensor view")
                    return self.stock(*args, **kw)
                for kernel, values in bound:
                    kernel(*values)
                self.counts["replays"] += 1
                return None

            self.key = self.plan = self.owners = None
            # This is the actual stock call. Never retry an operation after
            # launches started; a failure in optional observation only drops the plan.
            try:
                result = self.trace(*args, **kw)
                owners = tuple(self.buffers.get(name) for name in self.names)
                try:
                    plan = _make_plan(self.calls, sources, owners, self.torch, stream.cuda_stream)
                    if plan is None:
                        self._stop("unrecognized compiled ABI")
                    else:
                        self.plan, self.owners, self.key = plan, owners, key
                        self.peak_bytes = max(self.peak_bytes, sum(b.nbytes for b in owners))
                        self.counts["primed"] += 1
                except _PROBE_ERRORS:
                    self._stop("unsupported workspace")
                return result
            finally:
                self.calls.clear()


def install(worker):
    global _adapter, _reason
    if _adapter is not None:
        return True
    from vllm_mach.stack import BASE_PACKAGES
    contract = worker._mach_2b_contract
    if (BASE_PACKAGES["vllm"] != "0.34.0" or not contract["production_throughput_profile"]
            or contract["ssm_dtype"] != "float32"):
        _reason = "outside profile"
        return False
    try:
        import torch
        from flashinfer import gdn_prefill as fi, utils
        stock = fi.cp_delta_rule_dsl_sm120
        files = ((stock.__code__.co_filename, SOURCE), (fi.__file__, PUBLIC), (utils.__file__, UTILS))
        if (torch.cuda.get_device_capability(worker.device) != (12, 0)
                or stock.__name__ != "cp_delta_rule_dsl_sm120"
                or any(sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in files)
                or stock.__globals__["_get_cache_buf"] is not utils._get_cache_buf):
            _reason = "different FlashInfer or GPU"
            return False
        device = torch.device("cuda", torch.cuda.current_device()) if worker.device.index is None else worker.device
        adapter = _Adapter(stock, utils._cache_buf, device, torch)
    except (ImportError, *_PROBE_ERRORS):
        _reason = "FlashInfer plan unavailable"
        return False
    @wraps(stock)
    def dispatch(*args, **kwargs):
        return adapter.dispatch(*args, **kwargs)
    fi.cp_delta_rule_dsl_sm120 = dispatch
    _adapter = adapter
    return True


def receipt():
    if _adapter is None:
        return {"installed": False, "reason": _reason}
    return {"installed": True, "reason": _adapter.reason, "source": SOURCE,
            "primed": _adapter.counts["primed"], "replays": _adapter.counts["replays"],
            "peak_bound_workspace_bytes": _adapter.peak_bytes, "extra_device_bytes": 0}
