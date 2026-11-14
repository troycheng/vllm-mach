"""Sampler metadata lifecycle without CUDA; installed-runtime test uses real 0.33 methods."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from vllm_mach.mxfp8.two_b import sampler_metadata as mod


def params(**changes):
    values = dict(temperature=.7, top_k=20, top_p=.95, presence_penalty=1.5,
                  seed=None, ignore_eos=True, n=1, min_tokens=0, min_p=0,
                  repetition_penalty=1, frequency_penalty=0)
    values.update(changes)
    return NS(**values)


class Bias: pass
class Penalties: pass
class BadWords: pass


def fixture():
    reqs = NS(req_id_to_index={}, index_to_req_id={})
    class Sampler:
        def __init__(self):
            self.logits_processors = [Bias(), Penalties(), BadWords()]
            self.penalties_state = self.logits_processors[1]
            self.thinking_budget_state = NS(enabled=False)
            self.trace_replay_state = None
            self.return_sampling_mask = self.compute_nans = self.use_fp64_gumbel = False
            self.adds = self.copies = 0
        def add_request(self, index, value): self.adds += 1
        def apply_staged_writes(self): self.copies += 1
    sampler = Sampler()
    return mod._RefreshState(sampler, reqs, (Bias, Penalties, BadWords)), sampler, reqs


def admit(state, reqs, name, index, value):
    reqs.req_id_to_index[name] = index
    reqs.index_to_req_id[index] = name
    return state.add(index, value)


class SamplerMetadataCPU(unittest.TestCase):
    def test_activation_admission_and_seeded_greedy_keep_original_refresh(self):
        state, sampler, reqs = fixture()
        state.refresh(); state.refresh()
        self.assertEqual((sampler.copies, state.skips), (1, 1))
        value = params(temperature=0, top_k=-1, top_p=1, presence_penalty=0, seed=123)
        self.assertTrue(mod._static_request(value))
        admit(state, reqs, "greedy", 0, value)
        state.refresh(); state.refresh()
        self.assertEqual((sampler.adds, sampler.copies, state.skips), (1, 2, 2))

    def test_unknown_removal_and_slot_reuse_recover(self):
        state, sampler, reqs = fixture()
        admit(state, reqs, "unknown", 0, params(thinking_token_budget=2))
        admit(state, reqs, "known", 1, params())
        state.refresh(); state.refresh()
        self.assertEqual((state.fallbacks, state.skips), (2, 0))
        reqs.req_id_to_index.pop("unknown"); reqs.index_to_req_id.pop(0)
        state.refresh(); state.refresh()
        self.assertEqual(state.skips, 1)
        admit(state, reqs, "reused", 0, params(presence_penalty=0))
        before = sampler.copies; state.refresh(); state.refresh()
        self.assertEqual((sampler.copies, state.skips), (before + 1, 2))

    def test_streaming_same_id_can_change_eligibility(self):
        state, sampler, reqs = fixture()
        admit(state, reqs, "stream", 0, params(structured_outputs=object())); state.refresh()
        admit(state, reqs, "stream", 0, params()); state.refresh(); state.refresh()
        self.assertEqual((state.fallbacks, state.skips), (1, 1))

    def test_feature_changes_use_original_and_flush_before_recovery(self):
        state, sampler, reqs = fixture()
        admit(state, reqs, "a", 0, params()); state.refresh()
        for field in ("return_sampling_mask", "compute_nans", "use_fp64_gumbel"):
            setattr(sampler, field, True); before = sampler.copies; state.refresh()
            setattr(sampler, field, False); state.refresh(); state.refresh()
            self.assertEqual(sampler.copies, before + 2)
        sampler.logits_processors.append(object()); before = sampler.copies; state.refresh()
        sampler.logits_processors.pop(); state.refresh(); state.refresh()
        self.assertEqual(sampler.copies, before + 2)
        sampler.penalties_state = object(); before = sampler.copies; state.refresh()
        sampler.penalties_state = sampler.logits_processors[1]; state.refresh(); state.refresh()
        self.assertEqual(sampler.copies, before + 2)

    def test_common_fastpath_does_not_walk_live_request_ids(self):
        state, sampler, reqs = fixture()
        class NoWalk(dict):
            def __iter__(self): raise AssertionError("walked stable live IDs")
        reqs.req_id_to_index = NoWalk()
        state.refresh(); state.refresh()
        self.assertEqual(state.skips, 1)

    def test_static_parameters_are_not_benchmark_literals(self):
        for value in (params(temperature=.37, top_k=7, top_p=.83, seed=91, ignore_eos=False),
                      params(temperature=1.2, top_k=-1, top_p=1, ignore_eos=False),
                      params(temperature=0, top_k=-1, top_p=1, seed=12, ignore_eos=False),
                      params(min_p=.2, repetition_penalty=1.2, frequency_penalty=.5,
                             presence_penalty=-.3)):
            self.assertTrue(mod._static_request(value))
        for field in ("structured_outputs", "thinking_token_budget", "extra_args", "logits_processors",
                      "logprobs", "prompt_logprobs", "logprob_token_ids", "logit_bias",
                      "bad_words_token_ids", "allowed_token_ids"):
            self.assertFalse(mod._static_request(params(**{field: object()})))

    def test_unselected_profiles_do_not_import_runtime(self):
        for contract in ({}, {"mode": "quality"},
                         {"mode": "production", "ssm_dtype": "float16", "hybrid_greedy_head": True},
                         {"mode": "quality", "ssm_dtype": "bfloat16", "hybrid_greedy_head": True},
                         {"mode": "production", "ssm_dtype": "bfloat16", "hybrid_greedy_head": False},
                         {"mode": "production", "ssm_dtype": "float32", "hybrid_greedy_head": False}):
            self.assertFalse(mod.install(NS(_mach_2b_contract=contract)))

    def test_fp32_and_bf16_admit_the_same_sampler_instance_lifecycle(self):
        for dtype in ("float32", "bfloat16"):
            with self.subTest(ssm_dtype=dtype):
                _, sampler, reqs = fixture()
                modules = {}
                for name, attribute, value in (
                    ("sampler", "Sampler", type(sampler)),
                    ("logit_bias", "LogitBiasState", Bias),
                    ("penalties", "PenaltiesState", Penalties),
                    ("bad_words", "BadWordsState", BadWords),
                ):
                    path = "vllm.v1.worker.gpu.sample." + name
                    module = ModuleType(path)
                    setattr(module, attribute, value)
                    modules[path] = module
                worker = NS(_mach_2b_contract={"mode": "production", "ssm_dtype": dtype,
                                              "hybrid_greedy_head": True},
                            model_runner=NS(sampler=sampler, req_states=reqs))
                # These module substitutions exist only for the guarded imports;
                # the same product refresh state runs for both admitted dtypes.
                with patch.dict(sys.modules, modules), patch.object(mod, "version", return_value="0.33.0"):
                    self.assertTrue(mod.install(worker))
                    self.assertFalse(mod.install(worker))
                sampler.apply_staged_writes()
                sampler.apply_staged_writes()
                self.assertEqual((sampler.copies, mod.inspect_worker(worker)["skips"]), (1, 1))


def test_selected_2b_sampler_metadata_actual_runtime():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("vllm") is None:
        import pytest
        pytest.skip("requires installed 0.33 runtime")
    import numpy as np
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu import buffer_utils as bu
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    import vllm.v1.worker.gpu.sample.penalties as penalties
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
    from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    size = 160
    def field(dtype):
        f = object.__new__(bu.UvaBackedTensor); f.dtype = dtype
        f.cpu = torch.zeros(size, dtype=dtype); f.np = f.cpu.numpy()
        pool = object.__new__(bu.UvaBufferPool)
        pool.size, pool.dtype, pool.max_concurrency, pool._curr = size, dtype, 2, 0
        pool._buffer_cls, pool._uva_bufs = bu.UvaBuffer, []
        for _ in range(2):
            b = object.__new__(bu.UvaBuffer); b.cpu = torch.zeros(size, dtype=dtype)
            b.np, b._uva = b.cpu.numpy(), b.cpu
            pool._uva_bufs.append(b)
        f.pool = pool; f.gpu = pool.copy_to_uva(f.np)
        return f
    def staged():
        x = object.__new__(bu.StagedWriteTensor)
        x._staged_write_indices, x._staged_write_starts = [], []
        x._staged_write_contents, x._staged_write_cu_lens, x._staged_write_max_len = [], [], 0
        x.gpu = torch.zeros((size, 8), dtype=torch.int32)
        return x
    def make():
        req = object.__new__(RequestState)
        req.max_num_reqs, req.device, req.num_speculative_steps = size, torch.device("cpu"), 0
        req.req_id_to_index, req.index_to_req_id, req.free_indices = {}, {}, list(range(size))
        req.max_seq_len = np.zeros(size, dtype=np.int32)
        req.prompt_len, req.prefill_len = field(torch.int32), field(torch.int32)
        req.total_len, req.all_token_ids, req.num_computed_tokens = staged(), staged(), staged()
        req.num_computed_prefill_tokens, req.num_computed_tokens_np = np.zeros(size, dtype=np.int32), np.zeros(size, dtype=np.int32)
        s = object.__new__(Sampler); s.req_states = req
        st = object.__new__(SamplingStates); st.vocab_size, st.max_num_reqs = 248320, size
        for name, dtype in (("temperature", torch.float32), ("top_p", torch.float32),
                            ("top_k", torch.int32), ("min_p", torch.float32), ("seeds", torch.int64)):
            setattr(st, name, field(dtype))
        st.top_k.np.fill(st.vocab_size); st.top_k.copy_to_uva()
        st.top_p.np.fill(1); st.top_p.copy_to_uva()
        st.seeds_set, st.num_logprobs = np.zeros(size, dtype=bool), np.full(size, -1, dtype=np.int32)
        s.sampling_states = st
        pen = object.__new__(penalties.PenaltiesState); pen.req_states, pen.device = req, req.device
        pen._new_penalties_reqs, pen.use_penalty = [], np.zeros(size, dtype=bool)
        pen.prompt_bin_mask, pen.output_bin_counts = torch.zeros((size, 1), dtype=torch.int32), torch.zeros((size, 8), dtype=torch.int32)
        for name in ("repetition_penalty", "frequency_penalty", "presence_penalty"): setattr(pen, name, field(torch.float32))
        pen.repetition_penalty.np.fill(1); pen.repetition_penalty.copy_to_uva()
        bias = object.__new__(LogitBiasState); bias.prompt_len_np, bias.use_logit_bias = req.prompt_len.np, np.zeros(size, dtype=bool)
        for name in ("num_allowed_token_ids", "num_logit_bias", "min_lens", "num_stop_token_ids", "restore_when_all_masked"): setattr(bias, name, field(torch.int32))
        for name in ("allowed_token_ids", "logit_bias_token_ids", "logit_bias", "stop_token_ids"): setattr(bias, name, staged())
        bad = object.__new__(BadWordsState); bad.num_bad_words = field(torch.int32)
        bad.bad_word_token_ids, bad.bad_word_offsets = staged(), staged()
        lp = object.__new__(LogprobTokenIdsState); lp.num_token_ids, lp.token_ids, lp.max_num_reqs = field(torch.int32), staged(), size
        think = object.__new__(ThinkingBudgetState); think.enabled = False
        s.logits_processors, s.penalties_state = [bias, pen, bad], pen
        s.logprob_token_ids_state, s.thinking_budget_state, s.trace_replay_state = lp, think, None
        s.needs_logits_processing = np.zeros(size, dtype=bool)
        s.return_sampling_mask = s.compute_nans = s.use_fp64_gumbel = False
        return s, req
    def all_fields(s):
        groups = [(s.sampling_states, ("temperature", "top_p", "top_k", "min_p", "seeds")),
                  (s.penalties_state, ("repetition_penalty", "frequency_penalty", "presence_penalty")),
                  (s.logits_processors[0], ("num_allowed_token_ids", "num_logit_bias", "min_lens", "num_stop_token_ids", "restore_when_all_masked")),
                  (s.logits_processors[2], ("num_bad_words",)), (s.logprob_token_ids_state, ("num_token_ids",))]
        return [getattr(obj, name) for obj, names in groups for name in names]
    original, reference_reqs = make(); selected, selected_reqs = make()
    worker = NS(_mach_2b_contract={"mode": "production", "ssm_dtype": "float32", "hybrid_greedy_head": True},
                model_runner=NS(sampler=selected, req_states=selected_reqs))
    assert mod.install(worker) and not mod.install(worker)
    def admit_real(s, reqs, name, p):
        reqs.add_request(name, 3, [1, 2, 3], 0, 32)
        s.add_request(req_idx=reqs.req_id_to_index[name], sampling_params=p)
    removal_order = []
    def remove_real(reqs, name, preempted):
        slot = reqs.req_id_to_index[name]
        def model_remove(r):
            assert reqs.req_id_to_index[r] == slot
            removal_order.append((r, slot, preempted))
        runner = NS(req_states=reqs, model_state=NS(remove_request=model_remove), pooling_runner=None,
                    pp_handler=None, encoder_cache=None, prompt_logprobs_worker=None, lora_state=NS(remove_request=lambda r: None))
        runner._remove_request = lambda r: GPUModelRunner._remove_request(runner, r)
        GPUModelRunner.finish_requests(runner, NS(finished_req_ids=set() if preempted else {name}, preempted_req_ids={name} if preempted else set()))
        assert name not in reqs.req_id_to_index and slot not in reqs.index_to_req_id
        assert reqs.free_indices[-1] == slot
    histogram = selected.penalties_state.output_bin_counts
    pending = []
    with patch.object(penalties, "async_tensor_h2d", lambda x, **kw: torch.tensor(x, dtype=kw["dtype"])), \
            patch.object(penalties, "bincount", lambda idx, a, b, c, d, counts, n: counts.index_fill_(0, idx, 0)):
        for step in range(24):
            if len(pending) == 2: pending.pop(0)
            if step in (0, 3, 4, 8, 12, 16, 17):
                name = "request" + str(step)
                p = SamplingParams(temperature=0 if step % 3 == 0 else .37, top_k=-1 if step % 3 == 0 else 7,
                                   top_p=1 if step % 3 == 0 else .83, presence_penalty=0 if step % 3 == 0 else -.3,
                                   repetition_penalty=1.2, frequency_penalty=.5, min_p=.1 if step % 3 else 0,
                                   ignore_eos=step % 2 == 0, seed=123 if step % 2 else None,
                                   thinking_token_budget=2 if step == 4 else None)
                rng = np.random.get_state(); admit_real(original, reference_reqs, name, p)
                after = np.random.get_state(); np.random.set_state(rng); admit_real(selected, selected_reqs, name, p)
                assert all(np.array_equal(a, b) for a, b in zip(after, np.random.get_state()))
            if step in (6, 10):
                name = "request4" if step == 6 else "request3"
                remove_real(reference_reqs, name, step == 6); remove_real(selected_reqs, name, step == 6)
            guards = [(view, view.clone()) for group in pending for view in group]
            original.apply_staged_writes(); selected.apply_staged_writes()
            assert all(torch.equal(view, value) for view, value in guards)
            for a, b in zip(all_fields(original), all_fields(selected)):
                assert a.np.tobytes() == b.np.tobytes() and torch.equal(a.gpu, b.gpu)
                indices = list(reversed(list(selected_reqs.req_id_to_index.values())))
                assert torch.equal(a.gpu[indices], b.gpu[indices])
            histogram[0, 0] += 1
            assert histogram is selected.penalties_state.output_bin_counts
            pending.append([f.gpu for f in all_fields(selected)])
    state = getattr(worker, mod._STATE)
    assert state.skips > 0 and state.fallbacks > 0 and not state.unsupported
    assert state.admissions == 7 and selected_reqs.req_id_to_index == reference_reqs.req_id_to_index
    assert [r[0] for r in removal_order] == ["request4", "request4", "request3", "request3"]
    assert not torch.cuda.is_initialized()
