# SPDX-License-Identifier: Apache-2.0
"""Refresh stable per-request sampler metadata only after an admission."""
from functools import wraps
from importlib.metadata import version
from types import MethodType

_STATE = "_mach_2b_sampler_metadata"


def _static_request(params):
    # Scalar sampling parameters, seeds and EOS behavior have admission-only
    # writers. Sparse GPU payloads and grammar/reasoning/extensions are kept
    # on the original path until their complete consumer lifecycle is qualified.
    return (getattr(params, "min_tokens", 0) == 0
            and all(getattr(params, name, None) is None for name in
                    ("structured_outputs", "thinking_token_budget", "logprobs",
                     "prompt_logprobs", "logprob_token_ids"))
            and not any(getattr(params, name, None) for name in
                        ("extra_args", "logits_processors", "logit_bias",
                         "allowed_token_ids", "bad_words_token_ids")))


class _RefreshState:
    def __init__(self, sampler, req_states, processor_types):
        self.sampler, self.req_states = sampler, req_states
        self.processor_types = processor_types
        self.processors = tuple(sampler.logits_processors)
        self.original_add = sampler.add_request
        self.original_refresh = sampler.apply_staged_writes
        self.dirty = True
        self.unmapped = False
        self.unsupported = set()
        self.admissions = self.refreshes = self.skips = self.fallbacks = 0

    def supported(self):
        s, expected = self.sampler, self.processors
        current = s.logits_processors
        return (len(current) == 3
                and tuple(map(type, current)) == self.processor_types
                and all(a is b for a, b in zip(current, expected))
                and current[1] is s.penalties_state
                and not s.thinking_budget_state.enabled
                and s.trace_replay_state is None and not s.return_sampling_mask
                and not s.compute_nans and not s.use_fp64_gumbel)

    def add(self, req_idx, params):
        self.dirty = True
        result = self.original_add(req_idx, params)
        self.admissions += 1
        req_id = self.req_states.index_to_req_id.get(req_idx)
        if req_id is None:
            # Non-request/dummy admissions retain original refresh until a real admission.
            self.unmapped = True
        else:
            self.unmapped = False
            if _static_request(params):
                self.unsupported.discard(req_id)
            else:
                self.unsupported.add(req_id)
        return result

    def refresh(self):
        if self.unsupported:
            # Only the fallback path walks the live IDs. Stable request slots may be
            # removed, preempted, reordered or reused without another sampler callback.
            self.unsupported.intersection_update(self.req_states.req_id_to_index)
        eligible = not self.unsupported and not self.unmapped and self.supported()
        if eligible and not self.dirty:
            self.skips += 1
            return None
        result = self.original_refresh()
        self.refreshes += 1
        self.fallbacks += not eligible
        # Refresh once when an unsupported feature/request disappears as well.
        self.dirty = not eligible
        return result


def install(worker):
    contract = getattr(worker, "_mach_2b_contract", {})
    if (contract.get("mode") != "production"
            or contract.get("ssm_dtype") not in ("float32", "bfloat16")
            or not contract.get("hybrid_greedy_head")):
        return False
    if getattr(worker, _STATE, None) is not None:
        return False
    # The enclosing profile also checks the installed 0.33 sampler/states/runner sources.
    if version("vllm").split("+", 1)[0] != "0.33.0":
        raise RuntimeError("2B sampler metadata requires the selected vLLM runtime")
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
    from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
    runner = worker.model_runner
    sampler = runner.sampler
    if (type(sampler) is not Sampler
            or "add_request" in vars(sampler)
            or "apply_staged_writes" in vars(sampler)):
        return False
    state = _RefreshState(sampler, runner.req_states,
                          (LogitBiasState, PenaltiesState, BadWordsState))
    if not state.supported():
        return False

    @wraps(Sampler.add_request)
    def add(self, req_idx, sampling_params):
        return state.add(req_idx, sampling_params)

    @wraps(Sampler.apply_staged_writes)
    def refresh(self):
        return state.refresh()

    sampler.add_request = MethodType(add, sampler)
    sampler.apply_staged_writes = MethodType(refresh, sampler)
    setattr(worker, _STATE, state)
    return True


def inspect_worker(worker):
    state = getattr(worker, _STATE, None)
    if state is None:
        return {"installed": False}
    return {"installed": True, "admissions": state.admissions,
            "refreshes": state.refreshes, "skips": state.skips,
            "fallbacks": state.fallbacks,
            "unsupported_live": len(state.unsupported.intersection(
                state.req_states.req_id_to_index)), "dirty": state.dirty}
