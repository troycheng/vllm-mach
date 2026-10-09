# SPDX-License-Identifier: Apache-2.0
"""Compose the qualified 2B native/GDN/BA/PW lifecycle in one GPU worker."""
from functools import wraps
from importlib.metadata import version
import json
import os
from pathlib import Path

from ..profile import DEPENDENCIES, check_runtime_sources

NAME = "qwen35-2b-mxfp8-champion-v1"


def quality_contract(rows):
    if rows not in (4, 32, 64):
        raise ValueError("2B quality rows must be 4, 32 or 64")
    captures = [4] if rows == 4 else ([4, 32] if rows == 32 else [4, 32, 64])
    return {"max_model_len": 1024, "max_num_seqs": rows,
            "max_num_batched_tokens": max(8192, rows * 256),
            "kv_cache_memory_bytes": 4 * 2**30,
            "compilation_config": {"cudagraph_capture_sizes": captures,
                                   "max_cudagraph_capture_size": rows,
                                   "cudagraph_mode": "FULL_AND_PIECEWISE"}}


def run_contract():
    from .graph_policy import compilation_settings
    mode = os.environ.get("VLLM_MACH_MXFP8_MODE", "production")
    row = os.environ.get("VLLM_MACH_MXFP8_QUALITY_ROWS")
    if mode == "quality" and row in ("4", "32", "64"):
        rows = int(row)
        geometry = quality_contract(rows)
    elif mode == "production" and row is None:
        rows = None
        geometry = {"max_model_len": 16384, "max_num_seqs": 160,
                    "max_num_batched_tokens": 2048,
                    "kv_cache_memory_bytes": 19 * 2**30,
                    "compilation_config": compilation_settings()}
    else:
        raise RuntimeError("Select production without QUALITY_ROWS, or quality with rows4/32/64")
    if os.environ.get("VLLM_MACH_2B_PINNED_IDS", "1") != "1":
        raise RuntimeError("The 2B champion requires pinned reset-ID uploads")
    return {"mode": mode, "quality_rows": rows, **geometry,
            "attention_backend": "FLASH_ATTN", "kv_dtype": "bfloat16",
            "head_dtype": "bfloat16", "ssm_dtype": "float32",
            "production_throughput_profile": mode == "production"}


def check_environment():
    actual = {name: version(name) for name in DEPENDENCIES}
    for name, required in DEPENDENCIES.items():
        if actual[name].split("+", 1)[0] != required:
            raise RuntimeError(f"{NAME} requires {name}=={required}, found {actual[name]}")
    return actual


def _worker(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if worker is None:
        raise RuntimeError("2B worker is not initialized")
    return worker


def _check_geometry(worker, contract):
    cfg = worker.vllm_config
    text = cfg.model_config.hf_text_config
    expected = {"hidden_size": 2048, "intermediate_size": 6144,
                "num_hidden_layers": 24, "num_attention_heads": 8,
                "num_key_value_heads": 2, "head_dim": 256}
    if any(getattr(text, k, None) != v for k, v in expected.items()):
        raise RuntimeError("The 2B profile requires Qwen3.5-2B text geometry")
    backend = cfg.attention_config.backend
    if getattr(backend, "name", str(backend)) != "FLASH_ATTN":
        raise RuntimeError("The 2B profile requires FlashAttention 2")
    if (cfg.parallel_config.tensor_parallel_size != 1
            or cfg.parallel_config.pipeline_parallel_size != 1
            or cfg.parallel_config.data_parallel_size != 1
            or cfg.lora_config is not None
            or cfg.scheduler_config.max_num_batched_tokens != contract["max_num_batched_tokens"]
            or cfg.cache_config.cache_dtype not in ("auto", "bfloat16")
            or os.environ.get("VLLM_FLASH_ATTN_VERSION") != "2"):
        raise RuntimeError("2B TP1, BF16 KV, fixed prefill and non-LoRA contract changed")


def inspect_worker(worker):
    worker = _worker(worker)
    from . import ba, gdn, graph_policy
    from ..worker import verify_worker_execution
    import torch
    contract = worker._mach_2b_contract
    if contract != run_contract():
        raise RuntimeError("2B profile contract changed after worker initialization")
    _check_geometry(worker, contract)
    graph = graph_policy.verify_capture(worker)
    native = verify_worker_execution(worker, require_capture=True)
    state = gdn.inspect_worker(worker)
    required = set(contract["compilation_config"]["cudagraph_capture_sizes"])
    if not state["ready"] or state["layer_count"] != 18 or not state["pinned_ids_enabled"]:
        raise RuntimeError("2B GDN startup is incomplete")
    for name in state["layers"]:
        if any(state["capture_construction"].get(f"{name}|m{m}", 0) == 0
               for m in set(state["eligible_rows"]) & required):
            raise RuntimeError(f"Missing 2B GDN graph capture: {name}")
    projection = ba.verify_capture(worker, required_rows=(4, 8) if contract["mode"] == "production" else (4,))
    heads = [m for name, m in worker.get_model().named_modules() if name.endswith("lm_head")]
    if (len(heads) != 1 or heads[0].weight.dtype != torch.bfloat16
            or tuple(heads[0].weight.shape) != (248320, 2048)):
        raise RuntimeError("The 2B profile requires the complete BF16 vocabulary head")
    return {"profile": NAME, "pid": os.getpid(), **contract,
            "versions": check_environment(),
            "runtime_sources": worker._mach_2b_runtime_sources,
            "native": native, "gdn": state, "ba": projection, "graph": graph,
            "head": {"dtype": "bfloat16", "shape": [248320, 2048]},
            "counter_scope": "Python execution and CUDA graph construction; not replay counts"}


def write_receipt(worker, phase):
    worker = _worker(worker)
    receipt = {"phase": phase, **inspect_worker(worker)}
    directory = Path(os.environ["VLLM_MACH_MXFP8_RUN_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"worker-{os.getpid()}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)
    return receipt


def install_worker_hook():
    if os.environ.get("VLLM_MACH_PROFILE") != NAME:
        raise RuntimeError("Explicit 2B champion profile selection required")
    check_environment()
    contract = run_contract()
    sources = check_runtime_sources()
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker, "_mach_2b_hook", False):
        if Worker._mach_2b_hook_contract != contract:
            raise RuntimeError("2B profile changed after registration")
        return False
    if any(getattr(Worker, name, False) for name in
           ("_mach_mxfp8_native_hook", "_mach_mxfp8_champion_hook", "_two_b_gdn_ordered_worker_hook", "_two_b_ba_hook")):
        raise RuntimeError("Another MXFP8 worker profile is already installed")
    from . import ba, gdn, graph_policy
    from ..worker import install_worker_backend
    if contract["mode"] == "production":
        graph_policy.install_dispatch()
    original_init, original_load = Worker.init_device, Worker.load_model
    original_cache = Worker.initialize_from_config
    original_compile, original_execute = Worker.compile_or_warm_up_model, Worker.execute_model

    @wraps(original_init)
    def init(self, *args, **kwargs):
        self._mach_2b_contract = dict(contract)
        self._mach_2b_runtime_sources = sources
        self._mach_2b_ready = False
        self._mach_2b_steps = 0
        result = original_init(self, *args, **kwargs)
        _check_geometry(self, contract)
        install_worker_backend(self)
        gdn.install_runtime(self)
        ba.install(self)
        return result

    @wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        gdn.prepare_layers(self)
        return result

    @wraps(original_cache)
    def cache(self, *args, **kwargs):
        from .gdn import worker as state
        if state._READY or state._CACHE_INITIALIZED:
            raise RuntimeError("2B GDN does not support live cache reallocation")
        result = original_cache(self, *args, **kwargs)
        gdn.initialize_cache(self)
        return result

    @wraps(original_compile)
    def compile_model(self, *args, **kwargs):
        if run_contract() != self._mach_2b_contract:
            raise RuntimeError("2B profile changed before compilation")
        gdn.prepare_pools(self)
        result = original_compile(self, *args, **kwargs)
        write_receipt(self, "compiled")
        self._mach_2b_ready = True
        return result

    @wraps(original_execute)
    def execute(self, *args, **kwargs):
        result = original_execute(self, *args, **kwargs)
        if getattr(self, "_mach_2b_ready", False):
            self._mach_2b_steps += 1
            if self._mach_2b_steps == 1:
                write_receipt(self, "serving")
        return result

    Worker.init_device, Worker.load_model = init, load
    Worker.initialize_from_config = cache
    Worker.compile_or_warm_up_model, Worker.execute_model = compile_model, execute
    Worker._mach_2b_hook = True
    Worker._mach_2b_hook_contract = contract
    return True
