"""Experimental reuse of the compact greedy head with full-score fallbacks."""
import dataclasses
import types

_STATE = '_mach_4b_compact_head'


def install(worker):
    import torch
    from . import head as full
    from .two_b.hybrid_head import _wrap_sample
    from ..mxfp6 import hybrid_nvfp4_lm_head as nv
    from ..mxfp6.argmax_triton import indexed_argmax_triton
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    if hasattr(worker, _STATE):
        raise RuntimeError('Compact head already installed')
    runner = worker.model_runner
    if runner.batch_sharder is not None or runner.lora_config is not None or runner.speculative_config is not None:
        raise RuntimeError('Compact head requires ordinary TP1 sampling')
    original = getattr(worker, full._STATE)['fp4']
    state = dataclasses.replace(original, candidates=128, max_rows=128, can_use_failure_counts={})
    model = runner.model
    nv.warmup_hybrid_nvfp4_lm_head_kernels(state, model.lm_head.weight, tp_size=1)
    stats = dict(calls=0, sampling_fallback_calls=0, head_fallback_calls=0,
                 hybrid_success_calls=0, hybrid_success_rows=0, calls_by_rows={})

    def output(sampler, token_ids, batch):
        sampled, rejected = get_num_sampled_and_rejected(
            batch.seq_lens.new_ones(batch.num_reqs), batch.seq_lens,
            batch.cu_num_logits, batch.idx_mapping, sampler.req_states.prefill_len.gpu)
        return SamplerOutput(sampled_token_ids=token_ids.to(torch.int64).view(-1, 1),
                             logprobs_tensors=None, num_nans=None,
                             num_sampled=sampled, num_rejected=rejected)

    _wrap_sample(runner, model.lm_head, model.logits_processor, state, torch.bfloat16,
                 indexed_argmax_triton, output, stats)
    setattr(worker, _STATE, dict(state=state, stats=stats))
    return state


def install_audit(worker, state):
    """Untimed diagnostic: compare proposals to the unchanged full head."""
    import torch
    from ..mxfp6.argmax_triton import indexed_argmax_triton
    processor = worker.model_runner.model.logits_processor
    original = processor._apply_head
    counts = torch.zeros(3, device=worker.device, dtype=torch.int64)
    margin = torch.zeros((), device=worker.device, dtype=torch.float32)

    def apply(self, lm_head, hidden, embedding_bias):
        logits = original(lm_head, hidden, embedding_bias)
        coarse = state.coarse_logits(hidden, None)
        candidates = state.select_candidates(coarse)
        refined = state.refine_logits(hidden, lm_head.weight, candidates, None)
        _, proposed = indexed_argmax_triton(refined, candidates, index_offset=0)
        proposed = proposed.flatten().long()
        target = logits.argmax(dim=-1)
        row = torch.arange(target.numel(), device=hidden.device)
        loss = (logits[row, target].float() - logits[row, proposed].float()).clamp_min(0)
        counts[0].add_(target.numel())
        counts[1].add_((target != proposed).sum())
        counts[2].add_((loss > 0).sum())
        margin.copy_(torch.maximum(margin, loss.max()))
        return logits

    processor._apply_head = types.MethodType(apply, processor)
    worker._mach_head_audit = (counts, margin)


def inspect(worker):
    result = {}
    installed = getattr(worker, _STATE, None)
    if installed is not None:
        result.update(candidates=128, max_rows=128, shares_packed_weights=True,
                      stats={k:dict(v) if isinstance(v,dict) else v for k,v in installed['stats'].items()})
    if hasattr(worker, '_mach_head_audit'):
        counts, margin = worker._mach_head_audit
        rows, different, strict = counts.cpu().tolist()
        result['audit'] = dict(rows=rows, different_argmax=different, strict_score_losses=strict,
                               max_logit_loss=float(margin))
    return result
