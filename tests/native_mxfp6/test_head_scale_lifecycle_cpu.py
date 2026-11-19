"""Execute the installed Worker's post-capture head-scale gate without CUDA."""
from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
from enum import Enum
import hashlib
from importlib import metadata
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from vllm_mach.mxfp6 import install as installer
from vllm_mach.mxfp8.install import apply_profile_patch


class _GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL_AND_PIECEWISE = 2


class _CompilationMode(Enum):
    NONE = 0
    VLLM_COMPILE = 1


def _installed_method():
    manifest = installer.load_manifest()["namespaces"]["vllm"]
    name = "vllm/v1/worker/gpu_worker.py"
    if os.getenv("MACH_TEST_RUNTIME_PROFILE") == "native":
        source = metadata.distribution("vllm").locate_file(name)
        installed = Path(source).read_bytes()
        assert hashlib.sha256(installed).hexdigest() == manifest["files"][name]["installed_sha256"]
    else:
        source_root = os.getenv("VLLM_MACH_TEST_UPSTREAM_SOURCE")
        if not source_root:
            pytest.skip("Set VLLM_MACH_TEST_UPSTREAM_SOURCE to the pinned vLLM tree")
        before = {path: (Path(source_root) / path).read_bytes() for path in manifest["files"]}
        after = apply_profile_patch(before, (installer.profile_directory(installer.PROFILE) / manifest["patch"]).read_bytes(),
                                    manifest["shared_patch"])
        installed = after[name]
        assert hashlib.sha256(installed).hexdigest() == manifest["files"][name]["installed_sha256"]
    tree = ast.parse(installed)
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker")
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_compile_or_warm_up_model")
    # Compile the actual staged method body; only its external module globals are stubbed.
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    events = []
    globals_ = dict(CompilationTimes=lambda **kwargs: kwargs, CompilationMode=_CompilationMode,
                    CUDAGraphMode=_GraphMode, has_compiled_submodule=lambda _: False,
                    kernel_warmup=lambda _: events.append("kernel_warmup"),
                    warmup_kernels=lambda *args: events.append("warmup_kernels"),
                    set_random_seed=lambda *args: events.append("seed"),
                    freeze_gc_heap=lambda: events.append("freeze_gc"),
                    maybe_attach_gc_debug_callback=lambda: events.append("gc_callback"),
                    enable_gpu_sync_check=lambda: events.append("sync_gate"),
                    get_pp_group=lambda: NS(is_last_rank=True))
    exec(compile(module, "installed-vllm/v1/worker/gpu_worker.py", "exec"), globals_)
    return globals_["_compile_or_warm_up_model"], events


def _worker(events, *, v2=True, ssm="float16", model_type="qwen3_5_moe_text",
            hidden=2048, tp=2, graph=_GraphMode.FULL_AND_PIECEWISE, eager=False):
    @contextmanager
    def capture_context():
        events.append("capture_enter")
        try:
            yield
        finally:
            events.append("capture_exit")

    def capture_model():
        assert events[-1] == "capture_enter"
        events.append("capture_model")
        return 42

    runner = NS(lora_config=None, get_model=lambda: object(),
                maybe_remove_all_loras=lambda _: events.append("remove_loras"),
                capture_model=capture_model, _dummy_run=lambda **kwargs: (None, None),
                is_pooling_model=False,
                _dummy_sampler_run=lambda **kwargs: events.append("dummy_sampler"))
    compilation = NS(mode=_CompilationMode.VLLM_COMPILE, cudagraph_mode=graph,
                     backend="none", compilation_time=0, encoder_compilation_time=0)
    return NS(vllm_config=NS(compilation_config=compilation,
                             kernel_config=NS(enable_jit_warmup=False)),
              compilation_config=compilation, model_runner=runner,
              model_config=NS(enforce_eager=eager, hf_text_config=NS(model_type=model_type,
                                hidden_size=hidden), seed=0, max_model_len=16384,
                                quantization="quark",
                                architectures=["Qwen3_5MoeForConditionalGeneration"]
                                if model_type == "qwen3_5_moe_text" else ["Qwen3_5ForCausalLM"]),
              cache_config=NS(mamba_ssm_cache_dtype=ssm, kv_cache_memory_bytes=1,
                              enable_extensible_kv_cache=False),
              parallel_config=NS(tensor_parallel_size=tp, data_parallel_index=0),
              scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=2048),
              use_v2_model_runner=v2, _get_cudagraph_capture_context=capture_context,
              _maybe_activate_jit_monitor=lambda: events.append("jit_monitor"))


@pytest.mark.parametrize("change,selected", [
    ({}, True),
    ({"model_type": "qwen3_5_text"}, False),  # Dense
    ({"ssm": "float32"}, False),
    ({"model_type": "qwen3_5_text", "hidden": 1536}, False),  # 2B
    ({"tp": 1}, False),
    ({"graph": _GraphMode.PIECEWISE}, False),
    ({"eager": True}, False),
    ({"v2": False}, False),
])
def test_actual_installed_worker_lifecycle(monkeypatch, change, selected):
    method, events = _installed_method()
    target = "vllm_mach.mxfp6.head_scale"
    graph_target = "vllm_mach.mxfp6.gdn.prefill_graph"
    fake = ModuleType(target)
    fake.install = lambda worker: events.append("head_install")
    graph_fake = ModuleType(graph_target)
    graph_fake.install = lambda worker: events.append("prefill_install")
    monkeypatch.setitem(sys.modules, target, fake)
    monkeypatch.setitem(sys.modules, graph_target, graph_fake)
    original_import = builtins.__import__
    imports = []

    def observed_import(name, *args, **kwargs):
        if name in (target, graph_target):
            imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observed_import)
    v034 = installer.BASE_PACKAGES["vllm"] == "0.34.0"
    worker = _worker(events, **change)
    expected_result = {"language_model": 0, "encoder": 0}
    if v034:
        del worker.use_v2_model_runner
        expected_result["num_kv_blocks"] = None
        # The legacy runner case cannot occur in the 0.34 worker.
        if change.get("v2") is False:
            selected = True
    result = method(worker)
    assert result == expected_result
    assert events.count("capture_model") == (0 if change.get("eager") else 1)
    assert events.count("capture_exit") == (0 if change.get("eager") else 1)
    if selected:
        assert imports == [target, graph_target]
        assert events.index("capture_enter") < events.index("capture_model")
        assert events.index("capture_model") < events.index("capture_exit") < events.index("head_install") < events.index("prefill_install")
        assert events.count("head_install") == events.count("prefill_install") == 1
    else:
        assert not imports and "head_install" not in events and "prefill_install" not in events
