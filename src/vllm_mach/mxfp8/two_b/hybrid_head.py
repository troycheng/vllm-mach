# SPDX-License-Identifier: Apache-2.0
"""Opt-in K128 NVFP4 proposals and BF16 refinement for the 2B greedy head.

Install after worker warmup/capture. Stochastic sampling, processors and score
requests retain the original full BF16 path. This is an approximate greedy
head: indexed reduction can change equal-maximum BF16 token ties.
"""
from __future__ import annotations

import inspect
import types

_STATE = "_mach_2b_hybrid_head"
_HYBRID_STATES = ("_hybrid_nvfp4_lm_head_state", "_hybrid_mxfp4_lm_head_state",
                  "_hybrid_mxfp8_lm_head_state")


def _eligible(runner, batch, grammar_output):
    """Exact team greedy eligibility, with the isolated TP1 sharder guard."""
    sampler = runner.sampler
    if (runner.lora_config is not None or runner.speculative_config is not None
            or runner.batch_sharder is not None or grammar_output is not None
            or batch.is_prefilling_np.any() or sampler is None
            or batch.num_reqs == 0 or batch.num_draft_tokens != 0
            or sampler.compute_nans or sampler.return_sampling_mask
            or sampler.trace_replay_state is not None):
        return False
    indices = batch.idx_mapping_np
    return bool((sampler.sampling_states.temperature.np[indices] == 0.0).all()
                and not sampler.needs_logits_processing[indices].any()
                and sampler.get_logprobs_dims(indices) is None)


def _head_eligible(processor, head, hidden, state, bf16):
    shard = head.shard_indices
    active_vocab = shard.org_vocab_end_index - shard.org_vocab_start_index
    return (processor.head_dtype in (None, bf16)
            and processor.soft_cap is None and processor.scale == 1.0
            and shard.num_added_elements_padded == 0 and state.candidates <= 1024
            and state.can_use(hidden, bf16_weight=head.weight,
                              active_vocab_size=active_vocab, top_k=1))


def _wrap_sample(runner, head, processor, state, bf16, argmax, compact_output, stats):
    original = runner.sample

    def sample(self, hidden_states, input_batch, grammar_output):
        stats["calls"] += 1
        if not _eligible(self, input_batch, grammar_output):
            stats["sampling_fallback_calls"] += 1
            return original(hidden_states, input_batch, grammar_output)
        hidden = hidden_states[input_batch.logits_indices]
        if not _head_eligible(processor, head, hidden, state, bf16):
            stats["head_fallback_calls"] += 1
            return original(hidden_states, input_batch, grammar_output)
        # Preserve the exact qualified team operation order, dtype and selector.
        coarse = state.coarse_logits(hidden, None)
        active_vocab = (head.shard_indices.org_vocab_end_index
                        - head.shard_indices.org_vocab_start_index)
        coarse[:, active_vocab:] = -float("inf")
        candidates = state.select_candidates(coarse)
        refined = state.refine_logits(hidden, head.weight, candidates, None)
        _, token_ids = argmax(refined, candidates,
                             index_offset=head.shard_indices.org_vocab_start_index)
        output = compact_output(self.sampler, token_ids, input_batch)
        rows = int(hidden.shape[0])
        stats["hybrid_success_calls"] += 1
        stats["hybrid_success_rows"] += rows
        by_rows = stats["calls_by_rows"]
        by_rows[str(rows)] = by_rows.get(str(rows), 0) + 1
        return output, output.num_sampled, output.num_rejected

    runner.sample = types.MethodType(sample, runner)


def install(worker):
    """Install once on a captured TP1 worker without changing its BF16 head."""
    import torch
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm_mach.mxfp6 import hybrid_nvfp4_lm_head as nv
    from vllm_mach.mxfp6.argmax_triton import indexed_argmax_triton

    if hasattr(worker, _STATE):
        raise RuntimeError("2B hybrid head is already installed")
    runner = worker.model_runner
    model = runner.model
    head, processor = model.lm_head, model.logits_processor
    weight = head.weight
    if (worker.vllm_config.parallel_config.tensor_parallel_size != 1
            or runner.batch_sharder is not None or runner.lora_config is not None
            or runner.speculative_config is not None or runner.sampler is None
            or not runner.cudagraph_manager._graphs_captured
            or torch.cuda.is_current_stream_capturing()
            or weight.dtype != torch.bfloat16 or not weight.is_cuda
            or not weight.is_contiguous() or tuple(weight.shape) != (248320, 2048)
            or weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
            or processor.org_vocab_size != 248320
            or processor.head_dtype not in (None, torch.bfloat16)
            or processor.soft_cap is not None or processor.scale != 1.0
            or head.shard_indices.org_vocab_start_index != 0
            or head.shard_indices.org_vocab_end_index != 248320
            or head.shard_indices.num_added_elements_padded != 0
            or any(getattr(head, name, None) is not None for name in _HYBRID_STATES)):
        raise RuntimeError("Expected captured TP1 2B worker with untouched tied BF16 head")
    if tuple(inspect.signature(runner.sample).parameters) != (
            "hidden_states", "input_batch", "grammar_output"):
        raise RuntimeError("Runner.sample signature changed")

    # The alias shares storage but keeps generic preparation metadata off the
    # serving Parameter and module. It is retained for the sidecar lifetime.
    detached = torch.nn.Module()
    detached.register_parameter("weight", torch.nn.Parameter(weight.detach(), requires_grad=False))
    if not nv.prepare_hybrid_nvfp4_lm_head(detached, candidates=128):
        raise RuntimeError("NVFP4 b12x head preparation is unavailable")
    state = nv.get_hybrid_nvfp4_lm_head(detached)
    if (state is None or state.candidates != 128 or state.backend != "b12x"
            or state.input_size != 2048 or state.output_size != 248320):
        raise RuntimeError("NVFP4 K128 sidecar configuration changed")
    state.max_rows = 32
    nv.warmup_hybrid_nvfp4_lm_head_kernels(state, weight, tp_size=1)
    stats = dict(calls=0, sampling_fallback_calls=0, head_fallback_calls=0,
                 hybrid_success_calls=0, hybrid_success_rows=0, calls_by_rows={})

    def compact_output(sampler, token_ids, batch):
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            batch.seq_lens.new_ones(batch.num_reqs), batch.seq_lens,
            batch.cu_num_logits, batch.idx_mapping, sampler.req_states.prefill_len.gpu)
        return SamplerOutput(sampled_token_ids=token_ids.to(torch.int64).view(-1, 1),
                             logprobs_tensors=None, num_nans=None,
                             num_sampled=num_sampled, num_rejected=num_rejected)

    _wrap_sample(runner, head, processor, state, torch.bfloat16,
                 indexed_argmax_triton, compact_output, stats)
    setattr(worker, _STATE, dict(state=state, detached=detached, stats=stats))
    return inspect_worker(worker)


def inspect_worker(worker):
    """Return fresh JSON/RPC-safe primitive metadata and dispatch counters."""
    installed = getattr(worker, _STATE, None)
    if installed is None:
        return {"installed": False}
    state = installed["state"]
    stats = installed["stats"]
    return dict(installed=True, candidates=state.candidates, max_rows=state.max_rows,
                backend=state.backend, full_bf16_parameter=True,
                tie_guard=False, private_head_graph=False,
                **{k: dict(v) if isinstance(v, dict) else int(v) for k, v in stats.items()})


__all__ = ["install", "inspect_worker"]
