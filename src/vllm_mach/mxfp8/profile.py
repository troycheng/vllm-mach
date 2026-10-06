# SPDX-License-Identifier: Apache-2.0
"""Versioned Qwen3.5-4B profile with one ordered worker lifecycle.

Registration is CPU only. Every CUDA allocation occurs inside the GPU worker
after init_device; serving receipts describe actual installed paths.
"""
from __future__ import annotations

from functools import wraps
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path

NAME = "qwen35-4b-mxfp8-champion-v1"
DEPENDENCIES = {
    "vllm": "0.29.0", "torch": "2.13.0", "triton": "3.7.1",
    "flashinfer-python": "0.6.18", "flashinfer-cubin": "0.6.18",
    "nvidia-cutlass-dsl": "4.6.2", "cuda-bindings": "13.3.1",
    "b12x": "1.2.6", "mxfp6-sm120": "0.2.1",
}


def quality_contract(rows):
    """Accepted short-gold LLM geometry, independent of production capacity."""
    if rows not in (32, 64):
        raise ValueError("Quality rows must be 32 or 64")
    captures = [4, 32] if rows == 32 else [4, 32, 64]
    return {"dtype": "bfloat16", "tensor_parallel_size": 1,
            "max_model_len": 1024, "max_num_seqs": rows,
            "max_num_batched_tokens": rows * 256,
            "kv_cache_memory_bytes": (4 if rows == 32 else 19) * 2**30,
            "enable_chunked_prefill": True, "enable_prefix_caching": False,
            "attention_backend": "FLASHINFER", "language_model_only": True,
            "seed": 20260907, "logprobs_mode": "raw_logprobs", "disable_log_stats": True,
            "mamba_ssm_cache_dtype": "float32", "kv_cache_dtype": "fp8_e4m3",
            "quantization": "compressed-tensors",
            "kernel_config": {"linear_backend": "flashinfer_cutlass"},
            "compilation_config": {"cudagraph_capture_sizes": captures,
                                   "max_cudagraph_capture_size": rows,
                                   "cudagraph_mode": "FULL_AND_PIECEWISE"}}


def run_contract():
    """Production remains the default; quality is a fixed-M precision run."""
    mode = os.environ.get("VLLM_MACH_MXFP8_MODE", "production")
    if mode not in ("production", "quality"):
        raise RuntimeError(f"Unsupported MXFP8 profile mode: {mode}")
    rows = os.environ.get("VLLM_MACH_MXFP8_QUALITY_ROWS")
    if mode == "quality" and rows not in ("32", "64"):
        raise RuntimeError("Quality mode requires QUALITY_ROWS=32 or 64")
    if mode == "production" and rows is not None:
        raise RuntimeError("QUALITY_ROWS must be unset for production")
    geometry = quality_contract(int(rows)) if rows else {"kv_cache_memory_bytes": 19 * 2**30}
    return {"mode": mode, "quality_rows": int(rows) if rows else None, **geometry,
            "production_throughput_profile": mode == "production"}


def _quality_graph_receipt(worker, rows):
    from vllm.config.compilation import CUDAGraphMode
    manager = worker.model_runner.cudagraph_manager
    cfg = worker.vllm_config.compilation_config
    contract = quality_contract(rows)
    captures = contract["compilation_config"]["cudagraph_capture_sizes"]
    if manager is None or not manager._graphs_captured:
        raise RuntimeError("Quality FULL graph capture did not finish")
    # _capture_descs is the initialized plan, not proof of capture. The
    # accepted short-gold runs resolved to FULL decode + PIECEWISE mixed batches.
    planned = {mode.name: sorted(d.num_tokens for d in descs)
               for mode, descs in manager._capture_descs.items() if descs}
    expected = {"FULL": captures, "PIECEWISE": captures}
    if (planned != expected
            or cfg.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
            or manager.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
            or cfg.cudagraph_capture_sizes != captures
            or cfg.max_cudagraph_capture_size != rows
            or manager.use_breakable_cg):
        raise RuntimeError(f"Quality requires FULL_AND_PIECEWISE {captures}: {planned}")
    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    if (set(manager.graphs) != set(full_descs)
            or any(graph is None for graph in manager.graphs.values())):
        raise RuntimeError("Quality FULL graph capture incomplete")

    # PW graphs are owned by compiled subgraph wrappers, not manager.graphs.
    # Read vLLM's actual graph entries; an initialized entry without its CUDA
    # graph cannot establish capture. Ignore wrappers for other model configs.
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    pw_counts = {}
    for wrapper in list(CUDAGraphWrapper._all_instances):
        if (wrapper.vllm_config is not worker.vllm_config
                or wrapper.runtime_mode != CUDAGraphMode.PIECEWISE):
            continue
        for desc, entry in wrapper.concrete_cudagraph_entries.items():
            if entry.cudagraph is None:
                raise RuntimeError("Quality PIECEWISE graph capture incomplete")
            pw_counts[desc.num_tokens] = pw_counts.get(desc.num_tokens, 0) + 1
    if sorted(pw_counts) != captures:
        raise RuntimeError(f"Quality PIECEWISE graph capture incomplete: {pw_counts}")

    # Verify the actual dispatcher, including padding and uncaptured tails.
    # This also rejects accidental installation of the production PW2048 policy.
    for tokens in range(1, rows + 1):
        padded = next(size for size in captures if size >= tokens)
        for uniform, mode in ((1, CUDAGraphMode.FULL),
                              (None, CUDAGraphMode.PIECEWISE)):
            desc = manager.dispatch(num_reqs=tokens if uniform else 1,
                                    num_tokens=tokens, uniform_token_count=uniform,
                                    num_active_loras=0, max_query_len=1 if uniform else tokens)
            if desc.cg_mode != mode or desc.num_tokens != padded:
                raise RuntimeError(f"Quality graph dispatch changed at {tokens} tokens")
    tail = manager.dispatch(num_reqs=1, num_tokens=rows + 1,
                            uniform_token_count=None, num_active_loras=0)
    if tail.cg_mode != CUDAGraphMode.NONE or tail.num_tokens != rows + 1:
        raise RuntimeError("Quality uncaptured tail must run without graph padding")
    if (worker.vllm_config.cache_config.kv_cache_memory_bytes != contract["kv_cache_memory_bytes"]
            or worker.vllm_config.model_config.max_model_len != contract["max_model_len"]
            or worker.vllm_config.scheduler_config.max_num_seqs != contract["max_num_seqs"]
            or worker.vllm_config.scheduler_config.max_num_batched_tokens != contract["max_num_batched_tokens"]):
        raise RuntimeError(f"Quality M{rows} accepted capacity/scheduler geometry changed")
    return {"ready": True, "mode": "quality", "capture_sizes": captures,
            "full_sizes": captures, "piecewise_sizes": captures,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "piecewise_graph_counts": {str(m): count for m, count in sorted(pw_counts.items())},
            "production_throughput_profile": False}


def _verify_quality_capture(worker, rows):
    """Validate selected-M paths without claiming the full champion coverage."""
    from . import ba, compile_choices, dual, gdn, projection_parallel
    graph = _quality_graph_receipt(worker, rows)
    selected = dual.inspect_worker(worker)
    dual_rows = [m for m in graph["full_sizes"] if m in (32, 64)]
    for m in dual_rows:
        covered = {int(k) for k, count in selected["capture_layers"].get(str(m), {}).items() if count > 0}
        if not selected["installed"] or covered != set(range(24)):
            raise RuntimeError(f"Quality M{m} dual QKVZ capture incomplete: {covered}")
    if {int(k) for k, count in selected["mlp_capture_layers"].items()
                       if count > 0} != set(range(32)):
        raise RuntimeError("Quality M32 dual MLP capture incomplete")
    pairs = projection_parallel.inspect_worker(worker)
    for m in dual_rows:
        covered = {row["layer_id"] for row in pairs["counts"]
                   if row["phase"] == "capture" and row["m"] == m
                   and row["parallel"] and row["calls"] > 0}
        if not pairs["ready"] or len(pairs["pairs"]) != 24 or not pairs["aot_modules"] or covered != set(range(1, 24)):
            raise RuntimeError(f"Quality M{m} projection pair capture incomplete: {covered}")
    state = gdn.inspect_worker(worker)
    if (state["profile_mode"] != "quality" or state["quality_rows"] != rows
            or not state["ready"] or not state["cache_initialized"]
            or state["layer_count"] != 24):
        raise RuntimeError("Quality GDN runtime contract changed")
    for m in graph["full_sizes"]:
        covered = {key.rsplit("|m", 1)[0] for key, count in state["capture_construction"].items()
                   if key.endswith(f"|m{m}") and count > 0}
        if covered != set(state["layers"]):
            raise RuntimeError(f"Quality M{m} GDN capture incomplete: {covered}")
    ba.verify_capture(worker, required_rows=(4,))
    compile_choices.verify_coverage(require_complete=True)
    return graph


def check_environment():
    versions = {name: version(name) for name in DEPENDENCIES}
    for name, expected in DEPENDENCIES.items():
        if versions[name].split("+", 1)[0] != expected:
            raise RuntimeError(f"{NAME} requires {name}=={expected}, found {versions[name]}")
    return versions


def check_runtime_sources():
    """Gate parent and worker against the installed public runtime profile."""
    from .install import inspect_sources, load_manifest
    manifest = load_manifest()
    source = inspect_sources(Path(distribution("vllm").locate_file("")), manifest)
    if source["state"] != "installed":
        raise RuntimeError("Install the MXFP8 runtime source profile before serving")
    disabled = {**manifest["required_disabled_legacy_environment"],
                **manifest["required_disabled_compact_environment"]}
    actual = {name: os.environ.get(name) for name in disabled}
    if actual != disabled:
        raise RuntimeError(f"MXFP8 profile requires disabled legacy/compact paths: {actual}")
    return {**source, "disabled_environment": actual}


def _directory():
    path = Path(os.environ["VLLM_MACH_MXFP8_RUN_DIR"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_receipt(worker, phase):
    from . import ba, compile_choices, dual, gdn, graph_policy, head, kv
    from . import native_backend, projection_parallel

    receipt = {
        "profile": NAME, "phase": phase, "pid": os.getpid(),
        **worker._mach_mxfp8_contract,
        "versions": check_environment(),
        "runtime_sources": worker._mach_mxfp8_runtime_sources,
        "native": native_backend.inspect(worker),
        "dual": dual.inspect_worker(worker), "ba": ba.inspect_worker(worker),
        "gdn": gdn.inspect_worker(worker), "kv": kv.inspect_scales(worker),
        "graph": (_quality_graph_receipt(worker, worker._mach_mxfp8_contract["quality_rows"])
                  if worker._mach_mxfp8_contract["mode"] == "quality"
                  else graph_policy.inspect_worker(worker)),
        "projection_parallel": projection_parallel.inspect_worker(worker),
        "head": head.inspect_head(worker),
        "compile_choices": compile_choices.inspect(),
        "counter_scope": "Python execution and CUDA graph construction; graph replay is not counted",
    }
    target = _directory() / f"worker-{os.getpid()}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    return receipt


def install_worker_hook():
    """Install once in each parent/spawned process, before loading any model."""
    if os.environ.get("VLLM_MACH_PROFILE") != NAME:
        raise RuntimeError("Explicit champion profile selection required")
    check_environment()
    contract = run_contract()
    runtime_sources = check_runtime_sources()
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker, "_mach_mxfp8_champion_hook", False):
        if Worker._mach_mxfp8_champion_contract != contract:
            raise RuntimeError("MXFP8 profile contract changed after registration")
        return False
    if getattr(Worker, "_mach_mxfp8_native_hook", False):
        raise RuntimeError("Select the complete profile instead of stacking the standalone native hook")
    from . import compile_choices, graph_policy
    compile_choices.install()
    if contract["mode"] == "production":
        graph_policy.install_dispatch()

    original_init = Worker.init_device
    original_load = Worker.load_model
    original_cache = Worker.initialize_from_config
    original_compile = Worker.compile_or_warm_up_model
    original_execute = Worker.execute_model

    @wraps(original_init)
    def init(self, *args, **kwargs):
        self._mach_mxfp8_contract = dict(contract)
        self._mach_mxfp8_runtime_sources = runtime_sources
        self._mach_mxfp8_ready = False
        self._mach_mxfp8_steps = 0
        result = original_init(self, *args, **kwargs)
        from .worker import install_worker_backend
        from . import ba, dual, gdn
        install_worker_backend(self)
        dual.install(self)
        ba.install(self)
        gdn.install_runtime(self)
        return result

    @wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        from . import ba, dual, gdn, kv, projection_parallel
        kv.install_scales(self)
        dual.prepare_model(self)
        ba.prepare_model(self)
        gdn.prepare_layers(self)
        projection_parallel.prepare_model(
            self, cache_directory=_directory() / f"aot-{os.getpid()}"
        )
        return result

    @wraps(original_cache)
    def initialize_cache(self, *args, **kwargs):
        result = original_cache(self, *args, **kwargs)
        from . import gdn
        gdn.initialize_cache(self)
        return result

    @wraps(original_compile)
    def compile_model(self, *args, **kwargs):
        from . import ba, compile_choices, dual, gdn, graph_policy, head, kv
        from . import projection_parallel
        from .worker import verify_worker_execution
        if run_contract() != self._mach_mxfp8_contract:
            raise RuntimeError("MXFP8 profile contract changed before compilation")
        self._mach_mxfp8_ready = False
        self._mach_mxfp8_steps = 0
        gdn.prepare_pools(self)
        result = original_compile(self, *args, **kwargs)
        verify_worker_execution(self, require_capture=True)
        if contract["mode"] == "production":
            dual.verify_capture(self)
            ba.verify_capture(self)
            graph_policy.verify_capture(self)
            projection_parallel.verify_capture(self)
            compile_choices.verify_coverage()
        else:
            _verify_quality_capture(self, contract["quality_rows"])
        kv.inspect_scales(self)
        head.install_head(self)
        write_receipt(self, "compiled")
        self._mach_mxfp8_ready = True
        return result

    @wraps(original_execute)
    def execute(self, *args, **kwargs):
        result = original_execute(self, *args, **kwargs)
        # vLLM invokes this wrapper from compile_or_warm_up_model itself.
        # Those calls cannot publish a serving receipt before head installation.
        if not getattr(self, "_mach_mxfp8_ready", False):
            return result
        self._mach_mxfp8_steps += 1
        # Publish a real serving-path receipt once, outside timed GPU kernels.
        # Later snapshots are explicitly requested; do not add per-token I/O.
        if self._mach_mxfp8_steps == 1:
            write_receipt(self, "serving")
        return result

    Worker.init_device, Worker.load_model = init, load
    Worker.initialize_from_config = initialize_cache
    Worker.compile_or_warm_up_model = compile_model
    Worker.execute_model = execute
    Worker._mach_mxfp8_champion_hook = True
    Worker._mach_mxfp8_champion_contract = dict(contract)
    return True
