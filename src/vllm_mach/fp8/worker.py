# SPDX-License-Identifier: Apache-2.0
"""Install block-FP8 operators in the worker after device initialization."""
from __future__ import annotations

from functools import wraps
import json
import os
from pathlib import Path

from .profile import check_environment, check_runtime_sources, selected_features


def validate_worker_config(worker):
    cfg = worker.vllm_config
    parallel = cfg.parallel_config
    model = cfg.model_config
    text = model.hf_text_config
    expected = {"hidden_size": 2560, "intermediate_size": 9216,
                "num_hidden_layers": 32, "num_attention_heads": 16,
                "num_key_value_heads": 4, "head_dim": 256}
    if (getattr(text, "model_type", None) != "qwen3_5_text"
            or any(getattr(text, key, None) != value for key, value in expected.items())):
        raise RuntimeError("Block-FP8 profile requires the Qwen3.5-4B text geometry")
    if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1 or parallel.enable_dbo
            or cfg.speculative_config is not None):
        raise RuntimeError("Block-FP8 profile requires TP1/PP1 without speculation, DCP, or DBO")
    if str(model.dtype) != "torch.bfloat16" or model.quantization != "compressed-tensors":
        raise RuntimeError("Block-FP8 profile requires BF16 activations and compressed-tensors")
    if (cfg.cache_config.cache_dtype not in ("auto", "bfloat16")
            or cfg.cache_config.mamba_ssm_cache_dtype != "float32"):
        raise RuntimeError("Block-FP8 profile requires BF16 KV and FP32 SSM caches")
    offload = getattr(cfg, "offload_config", None)
    if (getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0)
            or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0)):
        raise RuntimeError("Block-FP8 profile does not support CPU weight offload")


def inspect(worker=None):
    from . import compile_choices
    features = selected_features()
    result = {"features": list(features), "compile_choices": compile_choices.inspect()}
    if "n64" in features or "ordered" in features:
        from . import linear
        result["linear"] = linear.inspect(worker)
    if "silu" in features:
        from . import activation
        result["activation"] = activation.inspect()
    if worker is not None:
        result["sources"] = worker._mach_fp8_sources
    return result


def write_receipt(worker):
    value = inspect(worker)
    directory = os.environ.get("VLLM_MACH_FP8_RUN_DIR")
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"worker-{os.getpid()}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, target)
    return value


def install_worker_hook():
    check_environment()
    sources = check_runtime_sources()
    features = selected_features()
    from . import compile_choices
    compile_choices.install()
    if "n64" in features or "ordered" in features:
        from . import linear
        linear.register()
    if "silu" in features:
        from . import activation
        activation.register()
    from vllm.v1.worker.gpu_worker import Worker
    old_features = getattr(Worker, "_mach_fp8_features", None)
    if old_features is not None:
        if old_features != features:
            raise RuntimeError("Block-FP8 feature selection changed after registration")
        return False
    original_init = Worker.init_device
    original_compile = Worker.compile_or_warm_up_model

    @wraps(original_init)
    def init(self, *args, **kwargs):
        validate_worker_config(self)
        result = original_init(self, *args, **kwargs)
        import torch
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Block-FP8 profile requires CUDA SM120")
        self._mach_fp8_sources = sources
        self._mach_fp8_compiled = False
        if "n64" in features or "ordered" in features:
            from . import linear
            linear.install(n64="n64" in features, ordered="ordered" in features)
        if "silu" in features:
            from . import activation
            activation.install()
        return result

    @wraps(original_compile)
    def compile_model(self, *args, **kwargs):
        if selected_features() != features:
            raise RuntimeError("Block-FP8 feature selection changed before compilation")
        result = original_compile(self, *args, **kwargs)
        compile_choices.verify_coverage()
        receipt = write_receipt(self)
        if "linear" in receipt:
            counts = receipt["linear"]["counts"]
            for route in ("n64", "ordered"):
                if route == "n64" and self.vllm_config.scheduler_config.max_num_seqs < 16:
                    continue
                if route in features and not counts.get(route + "_calls", 0):
                    raise RuntimeError(f"Selected block-FP8 {route} kernel never executed")
        if "silu" in features and not receipt["activation"]["counts"].get("native", 0):
            raise RuntimeError("Selected block-FP8 SiLU kernel never executed")
        self._mach_fp8_compiled = True
        return result

    def reject_weight_update(self, *args, **kwargs):
        raise RuntimeError(
            "Block-FP8 profile requires a worker restart to change model weights; "
            "captured graphs and layer-owned derived scales must be rebuilt"
        )

    Worker.init_device = init
    Worker.compile_or_warm_up_model = compile_model
    # These public worker RPCs can replace weight/scale storage after capture.
    # This profile has no online reload or weight-transfer contract.
    Worker.reload_weights = reject_weight_update
    Worker.start_weight_update = reject_weight_update
    Worker.update_weights = reject_weight_update
    Worker.init_weight_transfer_engine = reject_weight_update
    Worker._mach_fp8_features = features
    return True
