"""Only validated native TP2 MoE outputs may defer their final reduction."""

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config import CompilationMode

from vllm_mach.mxfp6.moe_ar_norm import defer_moe_allreduce, use_moe_ar_norm


@pytest.fixture
def config(monkeypatch):
    from vllm_mach.mxfp6 import moe_ar_norm

    monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "1")
    monkeypatch.setattr(moe_ar_norm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        moe_ar_norm.current_platform, "is_device_capability", lambda _: True
    )
    return NS(
        model_config=NS(
            dtype=torch.bfloat16,
            hf_text_config=NS(
                model_type="qwen3_5_moe_text",
                hidden_size=2048,
                num_experts=256,
                num_experts_per_tok=8,
                moe_intermediate_size=512,
                shared_expert_intermediate_size=512,
            ),
        ),
        parallel_config=NS(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            use_ubatching=False,
        ),
        quant_config=NS(
            get_name=lambda: "quark",
            quant_config={
                "global_quant_config": {
                    "weight": {"dtype": "fp6_e3m2"},
                    "input_tensors": {"dtype": "fp8_e4m3"},
                }
            },
        ),
        speculative_config=None,
        lora_config=None,
        compilation_config=NS(mode=CompilationMode.NONE),
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("tensor_parallel_size", 1),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("enable_expert_parallel", True),
        ("enable_eplb", True),
        ("use_sequence_parallel_moe", True),
        ("use_ubatching", True),
    ],
)
def test_parallel_admission(config, name, value):
    assert use_moe_ar_norm(config)
    setattr(config.parallel_config, name, value)
    assert not use_moe_ar_norm(config)


def test_model_quantization_and_request_admission(config, monkeypatch):
    assert use_moe_ar_norm(config)
    config.quant_config.quant_config["global_quant_config"]["weight"]["dtype"] = "fp8"
    assert not use_moe_ar_norm(config)
    config.quant_config.quant_config["global_quant_config"]["weight"]["dtype"] = (
        "fp6_e3m2"
    )
    config.speculative_config = object()
    assert not use_moe_ar_norm(config)
    config.speculative_config = None
    config.lora_config = object()
    assert not use_moe_ar_norm(config)
    config.lora_config = None
    monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "0")
    assert not use_moe_ar_norm(config)


def test_deferred_reduce_requires_native_runner_and_partitioned_shared_expert(
    monkeypatch,
):
    from vllm_mach.mxfp6 import moe

    class NativeRunner:
        def __init__(self):
            self.moe_config = NS(
                is_sequence_parallel=False,
                tp_size=2,
                ep_size=1,
                dp_size=1,
                skip_final_all_reduce=False,
            )
            self.routed_experts = NS()

    monkeypatch.setattr(moe, "Mxfp6Sm120MoERunner", NativeRunner)
    block = NS(experts=NativeRunner(), replicate_shared_expert=True)
    with pytest.raises(ValueError, match="local expert outputs"):
        defer_moe_allreduce(block)
    assert not block.experts.moe_config.skip_final_all_reduce
    block.replicate_shared_expert = False
    defer_moe_allreduce(block)
    assert block.experts.moe_config.skip_final_all_reduce
    assert block.experts.routed_experts._mach_fused_ar_norm
    block.experts = object()
    with pytest.raises(ValueError, match="native MXFP6"):
        defer_moe_allreduce(block)


def _worker_ar_norm(rank, port):
    import os

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
        tensor_model_parallel_all_reduce,
    )
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        _can_use_flashinfer,
        fused_allreduce_gemma_rms_norm,
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    os.environ["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] = "trtllm"
    os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
    os.environ["VLLM_SM120_LOSSLESS_PREFILL"] = "0"
    torch.cuda.set_device(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(2, rank, f"tcp://127.0.0.1:{port}", rank)
        ensure_model_parallel_initialized(2, 1)
        torch.manual_seed(100)
        norm = GemmaRMSNorm(2048, eps=1e-6).to(device=rank, dtype=torch.bfloat16)
        norm.weight.data.normal_(0, 0.1)
        for rows in (1, 4, 16, 24, 32, 3001, 4096):
            x = torch.empty(rows, 2048, device=rank, dtype=torch.bfloat16)
            residual = torch.empty_like(x)
            static = torch.empty_like(x)
            assert _can_use_flashinfer(x, 2)[0]
            x.normal_()
            residual.normal_()
            static.copy_(x)
            fused_allreduce_gemma_rms_norm(static, residual, norm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, updated = fused_allreduce_gemma_rms_norm(static, residual, norm)
            for step in range(3):
                # Residual and norm are replicated; rank-local partials differ.
                torch.manual_seed(1000 + rows + step)
                residual.normal_()
                torch.manual_seed(2000 + rows + step + rank)
                x.normal_()
                reference = norm(
                    tensor_model_parallel_all_reduce(x.clone()), residual.clone()
                )
                static.copy_(x)
                graph.replay()
                eager = fused_allreduce_gemma_rms_norm(x.clone(), residual, norm)
                for result, expected, direct in zip(
                    (out, updated), reference, eager, strict=True
                ):
                    assert torch.equal(result, direct)
                    relative = (
                        result.float() - expected.float()
                    ).norm() / expected.float().norm()
                    assert relative < 0.01
        cleanup_dist_env_and_memory()


def test_tp2_ar_norm_2048_changing_graphs():
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires two SM120 GPUs")
    from torch.multiprocessing import spawn
    from vllm.utils.network_utils import get_open_port

    spawn(_worker_ar_norm, args=(get_open_port(),), nprocs=2, join=True)
