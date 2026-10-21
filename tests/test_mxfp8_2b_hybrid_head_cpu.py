"""CPU checks of hybrid dispatch boundaries and untouched sampler fallbacks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/two_b/hybrid_head.py"


def load():
    spec = importlib.util.spec_from_file_location("hybrid_head_cpu", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Vector:
    def __init__(self, values): self.values = values
    def __getitem__(self, indices): return Vector([self.values[i] for i in indices])
    def __eq__(self, value): return Vector([v == value for v in self.values])
    def any(self): return any(self.values)
    def all(self): return all(self.values)


class Hidden:
    shape = (2, 2048)
    def __getitem__(self, key): return self


def fixture():
    events = []
    fallback = object()
    sampler = NS(compute_nans=False, return_sampling_mask=False, trace_replay_state=None,
                 sampling_states=NS(temperature=NS(np=Vector([0., 0.]))),
                 needs_logits_processing=Vector([False, False]),
                 get_logprobs_dims=lambda indices: None)
    def original(*args):
        events.append(("original", args))
        return fallback
    runner = NS(lora_config=None, speculative_config=None, batch_sharder=None,
                sampler=sampler, sample=original)
    batch = NS(is_prefilling_np=Vector([False, False]), num_reqs=2, num_draft_tokens=0,
               idx_mapping_np=[0, 1], logits_indices=[0, 1])
    head = NS(weight=object(), shard_indices=NS(org_vocab_start_index=0,
               org_vocab_end_index=248320, num_added_elements_padded=0))
    processor = NS(head_dtype=None, soft_cap=None, scale=1.)
    class Coarse:
        def __setitem__(self, key, value): events.append(("mask", key, value))
    state = NS(_head_graphs=None, output_size=248320, candidates=128, can_use=lambda *a, **k: True,
               coarse_logits=lambda *a: (events.append(("coarse", a)) or Coarse()),
               select_candidates=lambda *a: (events.append(("select", a)) or "indices"),
               refine_logits=lambda *a: (events.append(("refine", a)) or "refined"))
    stats = dict(calls=0, sampling_fallback_calls=0, head_fallback_calls=0,
                 hybrid_success_calls=0, hybrid_success_rows=0, calls_by_rows={})
    def argmax(*a, **k):
        events.append(("argmax", a, k))
        return "values", "tokens"
    output = NS(num_sampled="sampled", num_rejected="rejected")
    def compact(*a):
        events.append(("compact", a))
        return output
    return NS(events=events, fallback=fallback, runner=runner, batch=batch,
              head=head, processor=processor, state=state, stats=stats,
              argmax=argmax, compact=compact, output=output, hidden=Hidden())


class HybridHeadCPU(unittest.TestCase):
    def bind(self, f):
        load()._wrap_sample(f.runner, f.head, f.processor, f.state, "bf16",
                            f.argmax, f.compact, f.stats)

    def test_graph_replay_skips_eager_stages_and_preserves_fallback(self):
        f = fixture()
        f.state.replay_greedy = lambda *a: "graph_tokens"
        self.bind(f)
        f.runner.sample(f.hidden, f.batch, None)
        self.assertEqual([e[0] for e in f.events], ["compact"])
        self.assertEqual(f.events[0][1][1], "graph_tokens")
        f.events.clear()
        f.state.replay_greedy = lambda *a: None
        f.runner.sample(f.hidden, f.batch, None)
        self.assertEqual([e[0] for e in f.events],
                         ["coarse", "mask", "select", "refine", "argmax", "compact"])

    def test_import_without_gpu_packages(self):
        code = """import builtins, runpy, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'vllm', 'triton', 'numpy'):
        raise RuntimeError('GPU/runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
runpy.run_path(sys.argv[1])
"""
        subprocess.run([sys.executable, "-c", code, str(SOURCE)], check=True)

    def test_sampling_fallback_returns_original_identity_and_arguments(self):
        mutations = {
            "random": lambda f: setattr(f.runner.sampler.sampling_states.temperature, "np", Vector([0., .7])),
            "processor": lambda f: setattr(f.runner.sampler, "needs_logits_processing", Vector([False, True])),
            "logprobs_or_token_scores": lambda f: setattr(f.runner.sampler, "get_logprobs_dims", lambda i: (0, 1)),
            "prefill": lambda f: setattr(f.batch, "is_prefilling_np", Vector([False, True])),
            "draft": lambda f: setattr(f.batch, "num_draft_tokens", 1),
            "empty": lambda f: setattr(f.batch, "num_reqs", 0),
            "lora": lambda f: setattr(f.runner, "lora_config", object()),
            "speculation": lambda f: setattr(f.runner, "speculative_config", object()),
            "sharder": lambda f: setattr(f.runner, "batch_sharder", object()),
            "nan_diagnostics": lambda f: setattr(f.runner.sampler, "compute_nans", True),
            "sampling_mask": lambda f: setattr(f.runner.sampler, "return_sampling_mask", True),
            "replay": lambda f: setattr(f.runner.sampler, "trace_replay_state", object()),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                f = fixture(); mutate(f); self.bind(f)
                self.assertIs(f.runner.sample(f.hidden, f.batch, None), f.fallback)
                self.assertEqual(f.events, [("original", (f.hidden, f.batch, None))])
                self.assertEqual(f.stats["hybrid_success_calls"], 0)
        f = fixture(); self.bind(f); grammar = object()
        self.assertIs(f.runner.sample(f.hidden, f.batch, grammar), f.fallback)
        self.assertIs(f.events[0][1][2], grammar)

    def test_head_rejection_preserves_original_sampler(self):
        mutations = [lambda f: setattr(f.processor, "head_dtype", "fp32"),
                     lambda f: setattr(f.processor, "soft_cap", 20.),
                     lambda f: setattr(f.processor, "scale", .5),
                     lambda f: setattr(f.head.shard_indices, "num_added_elements_padded", 32),
                     lambda f: setattr(f.state, "can_use", lambda *a, **k: False)]
        for mutate in mutations:
            f = fixture(); mutate(f); self.bind(f)
            self.assertIs(f.runner.sample(f.hidden, f.batch, None), f.fallback)
            self.assertEqual(f.stats["head_fallback_calls"], 1)
            self.assertEqual([e[0] for e in f.events], ["original"])

    def test_eligible_seeded_greedy_executes_qualified_order_and_plain_receipt(self):
        f = fixture()
        f.runner.sampler.sampling_states.seeds = object()  # No seed guard for greedy.
        self.bind(f)
        result = f.runner.sample(f.hidden, f.batch, None)
        self.assertEqual(result, (f.output, "sampled", "rejected"))
        self.assertEqual([e[0] for e in f.events],
                         ["coarse", "mask", "select", "refine", "argmax", "compact"])
        self.assertEqual(f.events[3][1], (f.hidden, f.head.weight, "indices", None))
        self.assertEqual(f.events[4][2], {"index_offset": 0})
        f.state.max_rows, f.state.backend = 32, "b12x"
        worker = NS(_mach_2b_hybrid_head=dict(state=f.state, stats=f.stats))
        receipt = load().inspect_worker(worker)
        self.assertEqual(receipt["hybrid_success_rows"], 2)
        self.assertEqual(receipt["calls_by_rows"], {"2": 1})
        json.dumps(receipt)
        receipt["calls_by_rows"]["2"] = 999
        self.assertEqual(f.stats["calls_by_rows"], {"2": 1})


if __name__ == "__main__":
    unittest.main()
