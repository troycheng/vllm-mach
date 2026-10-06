# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-vocabulary NVFP4 head with K2048 BF16 score replacement.

Call install_head only after the worker's compile/warmup stage. The original
sampler and tied BF16 parameter stay in place. This module creates no private
CUDA graph and imports GPU dependencies only when installing or executing.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import types

VOCAB_SIZE = 248320
HIDDEN_SIZE = 2560
CANDIDATES = 2048
HEAD_SHA256 = "2a68153b498532801ab605bb03fe617c57b3e6a3ba019301fb70ada0c864a2f7"
PACKED_BYTES = 357580804
_STATE = "_mach_mxfp8_full_head_state"
_HYBRID_STATES = ("_hybrid_nvfp4_lm_head_state", "_hybrid_mxfp4_lm_head_state",
                  "_hybrid_mxfp8_lm_head_state")


def _tensor_sha(value):
    import torch
    digest = hashlib.sha256()
    flat = value.detach().contiguous().view(torch.uint8).reshape(-1)
    for chunk in flat.split(8 * 2**20):
        digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()


def full_logits(hidden, weight, state, *, coarse=None):
    """Return complete logits, chosen indices and their BF16 replacement scores.

    This retains the fixed K2048 sweep's operation order: coarse NVFP4 GEMM,
    torch.topk(sorted=False), existing indexed BF16 dot, then in-place scatter.
    It deliberately bypasses the compact head's candidate selection/sampler.
    """
    import torch
    if (state.candidates != CANDIDATES or hidden.ndim != 2
            or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous()
            or tuple(hidden.shape[1:]) != (HIDDEN_SIZE,) or not hidden.is_cuda
            or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
            or tuple(weight.shape) != (VOCAB_SIZE, HIDDEN_SIZE)
            or weight.device != hidden.device or state.output_size != VOCAB_SIZE
            or state.input_size != HIDDEN_SIZE):
        raise ValueError("Expected fixed CUDA BF16 full-vocabulary K2048 head inputs")
    coarse = state.coarse_logits(hidden, None) if coarse is None else coarse
    if (tuple(coarse.shape) != (hidden.shape[0], VOCAB_SIZE)
            or coarse.dtype != torch.bfloat16 or not coarse.is_contiguous()
            or coarse.device != hidden.device):
        raise ValueError("Coarse logits must be full-vocabulary BF16 on the input device")
    indices = torch.topk(coarse, CANDIDATES, dim=1, sorted=False).indices
    refined = state.refine_logits(hidden, weight, indices, None)
    if (refined.shape != indices.shape or refined.dtype != coarse.dtype
            or refined.device != coarse.device):
        raise RuntimeError("Indexed BF16 dot shape/dtype/device changed")
    coarse.scatter_(dim=1, index=indices.long(), src=refined)
    return coarse, indices, refined


def install_head(worker):
    """Install the full-logit boundary on this compiled worker, exactly once."""
    import torch
    from vllm_mach.mxfp6 import hybrid_nvfp4_lm_head as nv

    if hasattr(worker, _STATE):
        raise RuntimeError("K2048 full-vocabulary head is already installed")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Install the head after compilation, outside CUDA capture")
    if any(os.environ.get(name, "0") != "0" for name in
           ("VLLM_HYBRID_NVFP4_LM_HEAD", "VLLM_HYBRID_MXFP4_LM_HEAD",
            "VLLM_HYBRID_MXFP8_LM_HEAD")):
        raise RuntimeError("Stock compact hybrid head modes must be disabled")
    model = worker.model_runner.model
    head, processor = model.lm_head, model.logits_processor
    weight = head.weight
    embedding = model.model.embed_tokens.weight
    if (not weight.is_cuda or weight.dtype != torch.bfloat16
            or not weight.is_contiguous() or tuple(weight.shape) != (VOCAB_SIZE, HIDDEN_SIZE)
            or weight.data_ptr() != embedding.data_ptr()
            or processor.org_vocab_size != VOCAB_SIZE
            or processor.head_dtype not in (None, torch.bfloat16)
            or any(getattr(head, name, None) is not None for name in _HYBRID_STATES)):
        raise RuntimeError("Expected untouched tied BF16 full-vocabulary head")
    original = processor._apply_head
    parameters = tuple(inspect.signature(original).parameters)
    if parameters != ("lm_head", "hidden_states", "embedding_bias"):
        raise RuntimeError(f"LogitsProcessor._apply_head signature drift: {parameters}")
    # Preparation registers only a sidecar on a separate module. It shares the
    # original Parameter; the serving head never acquires compact hybrid state.
    detached = torch.nn.Module()
    detached.register_parameter("weight", weight)
    if not nv.prepare_hybrid_nvfp4_lm_head(detached, candidates=CANDIDATES):
        raise RuntimeError("NVFP4 b12x head preparation is unavailable")
    fp4 = detached._hybrid_nvfp4_lm_head_state
    if (fp4.candidates != CANDIDATES or fp4.output_size != VOCAB_SIZE
            or fp4.input_size != HIDDEN_SIZE or fp4.backend != "b12x"):
        raise RuntimeError("NVFP4 sidecar is not the K2048 b12x configuration")
    extra_bytes = sum(getattr(fp4, name).nbytes for name in ("weight", "scale", "global_scale"))
    if extra_bytes != PACKED_BYTES:
        raise RuntimeError("K2048 packed sidecar byte inventory changed")
    digest = _tensor_sha(weight)
    if digest != HEAD_SHA256:
        raise RuntimeError("Tied BF16 head identity changed after NVFP4 packing")
    if any(getattr(head, name, None) is not None for name in _HYBRID_STATES):
        raise RuntimeError("NVFP4 detached state leaked onto the serving head")
    state = {"head": head, "embedding": embedding, "processor": processor,
             "weight_ptr": weight.data_ptr(), "original": original, "detached": detached,
             "fp4": fp4, "extra_bytes": extra_bytes, "head_sha256": digest,
             "calls": {}, "capture_calls": 0}

    def apply(self, lm_head, hidden_states, embedding_bias):
        if (lm_head is not head or lm_head.weight.data_ptr() != state["weight_ptr"]
                or model.model.embed_tokens.weight.data_ptr() != state["weight_ptr"]):
            raise RuntimeError("Tied lm_head identity changed after installation")
        if embedding_bias is not None:
            raise RuntimeError("Full K2048 refinement requires a bias-free head")
        if torch.cuda.is_current_stream_capturing():
            state["capture_calls"] += 1
            raise RuntimeError("Full-vocabulary head must execute outside model CUDA capture")
        output = full_logits(hidden_states, weight, fp4)[0]
        m = int(hidden_states.shape[0])
        state["calls"][m] = state["calls"].get(m, 0) + 1
        return output

    processor._apply_head = types.MethodType(apply, processor)
    setattr(worker, _STATE, state)
    return inspect_head(worker)


def inspect_head(worker):
    """Return dispatch counts and invariants, without importing a sampler."""
    import torch
    state = getattr(worker, _STATE, None)
    if state is None:
        return {"installed": False}
    model = worker.model_runner.model
    head, weight = model.lm_head, model.lm_head.weight
    if (head is not state["head"] or weight.data_ptr() != state["weight_ptr"]
            or model.model.embed_tokens.weight.data_ptr() != state["weight_ptr"]
            or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (VOCAB_SIZE, HIDDEN_SIZE)
            or any(getattr(head, name, None) is not None for name in _HYBRID_STATES)):
        raise RuntimeError("Installed tied BF16 head contract changed")
    return {"installed": True, "backend": state["fp4"].backend,
            "head_shape": list(weight.shape), "head_dtype": str(weight.dtype),
            "head_sha256_after_pack": state["head_sha256"],
            "weight_ptr": state["weight_ptr"], "fp4_extra_bytes": state["extra_bytes"],
            "candidate_width": CANDIDATES, "full_vocab_logits": True,
            "selection_backend": "torch.topk(sorted=False)",
            "selected_scores_replaced": True, "stock_hybrid_state_on_real_head": False,
            "calls_by_rows": {str(m): n for m, n in state["calls"].items()},
            "refined_success_calls": sum(state["calls"].values()),
            "bf16_fallback_calls": 0, "capture_calls": state["capture_calls"],
            "private_head_graph": False,
            "org_vocab_size": state["processor"].org_vocab_size,
            "soft_cap": state["processor"].soft_cap, "scale": state["processor"].scale}


__all__ = ["install_head", "inspect_head", "full_logits"]
