"""Teacher ABI and token mapping without a model, CUDA or Torch dependency."""
import importlib.util
import importlib.metadata
import ast
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


class Array(list):
    def tolist(self):
        return list(self)


class Tensor:
    def __init__(self, values, shape=None):
        self.values = list(values)
        self.shape = shape or (len(values),)
        self.device = "cpu"
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return list(self.values)
    def numel(self):
        return len(self.values)


class TeacherDecodeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "cpu_teacher_decode", Path(__file__).parents[1] / "tools/quantization/teacher_decode.py"
        )
        self.teacher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.teacher)
        class Sampler:
            logprobs_mode = "raw_logprobs"
            def __init__(self):
                self.added, self.fallbacks = [], []
            def add_request(self, req_idx, sampling_params):
                self.added.append((req_idx, sampling_params))
            def sample(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np,
                       pos, input_ids, expanded_local_pos, seq_lens_upper_bound_np,
                       return_logprobs=False):
                self.fallbacks.append((seq_lens_upper_bound_np, return_logprobs, idx_mapping_np))
                return "original", logits
            def __call__(self, logits, input_batch):
                # Original caller keeps logits = processed_logits and compute_topk_scores(...).
                return self.sample(logits, None, None, input_batch.idx_mapping_np,
                                   input_batch.positions, None, None, input_batch.bounds,
                                   return_logprobs=True)
        self.original_add = inspect.signature(Sampler.add_request)
        self.original_sample = inspect.signature(Sampler.sample)
        self.sampler = Sampler()
        modules = {}
        for name in ["vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
                     "vllm.v1.worker.gpu.sample", "vllm.v1.worker.gpu.sample.sampler", "vllm.distributed"]:
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        modules["vllm.v1.worker.gpu.sample.sampler"].Sampler = Sampler
        modules["vllm.distributed"].get_tensor_model_parallel_rank = lambda: 0
        modules["torch"] = NS(int64="int64", tensor=lambda data, **kw: Tensor(data),
                              cuda=NS(is_current_stream_capturing=lambda: False))
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.environment = patch.dict(os.environ, K4O_QUALITY_ARM="m32")
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.worker = NS(model_runner=NS(sampler=self.sampler))
        self.teacher.install(self.worker)
        self.worker._k4o_teacher_shape_state["physical_rows"] = 4
        self.indices = Array([17, 2, 9, 4])
        self.prompt_lengths = [5, 9, 3, 7]
        for idx, length in zip(self.indices, self.prompt_lengths):
            self.add(idx, length, [100 + idx, 200 + idx, 300 + idx])

    def add(self, idx, length, forced):
        params = NS(extra_args={self.teacher.KEY: dict(id=str(idx), prompt_len=length,
                    forced_ids=forced, pad_id=forced[-1])}, temperature=0.0,
                    ignore_eos=True, logprobs=1, max_tokens=len(forced))
        self.sampler.add_request(idx, params)

    def call(self, offset):
        batch = NS(idx_mapping_np=self.indices,
                   positions=Tensor([n - 1 + offset for n in self.prompt_lengths]),
                   bounds=Array([999, 888, 777, 666]), input_ids=Tensor([0] * 4),
                   num_tokens=4, num_reqs=4)
        logits = Tensor([], shape=(4, 1000))
        forced, returned_logits = self.sampler(logits, batch)
        self.assertIs(returned_logits, logits)
        return forced.tolist()

    def test_signatures_match_v033_and_original_add_receives_req_idx(self):
        self.assertEqual(inspect.signature(type(self.sampler).add_request), self.original_add)
        self.assertEqual(inspect.signature(type(self.sampler).sample), self.original_sample)
        self.assertEqual([i for i, _ in self.sampler.added], list(self.indices))

    def test_signatures_match_installed_sampler_source_when_available(self):
        try:
            distribution = importlib.metadata.distribution("vllm")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("No installed vLLM; portable callback tests still run")
        path = Path(distribution.locate_file("vllm/v1/worker/gpu/sample/sampler.py"))
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sampler")
        for name in ("add_request", "sample"):
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            names = [n.arg for n in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            self.assertEqual(list(inspect.signature(getattr(type(self.sampler), name)).parameters), names)

    def test_first_token_gold_padding_and_observations_preserve_request_mapping(self):
        for offset, base in [(0, 100), (1, 200), (2, 300), (4, 300)]:
            self.assertEqual(self.call(offset), [base + idx for idx in self.indices])
        calls = self.teacher.observations(self.worker)["calls"]
        self.assertEqual([r["offset"] for r in calls], [0, 1, 2, 4])
        self.assertTrue(all(r["ids"] == [str(i) for i in self.indices] for r in calls))
        self.assertEqual(self.teacher.observations(self.worker)["calls"], [])

    def test_slot_reuse_replaces_previous_forced_sequence(self):
        self.add(17, 5, [501, 502, 503])
        self.assertEqual(self.call(0)[0], 501)
        self.assertEqual(self.call(1)[0], 502)

    def test_non_teacher_fallback_forwards_new_bound_unchanged(self):
        bounds = Array([1234])
        indices = Array([100])
        logits = Tensor([], shape=(1, 1000))
        result = self.sampler.sample(logits, None, None, indices, Tensor([1]), None, None,
                                     bounds, return_logprobs=True)
        self.assertEqual(result, ("original", logits))
        self.assertIs(self.sampler.fallbacks[-1][0], bounds)
        self.assertIs(self.sampler.fallbacks[-1][2], indices)
        self.assertTrue(self.sampler.fallbacks[-1][1])

    def test_mixed_teacher_and_non_teacher_batch_rejected(self):
        with self.assertRaises(AssertionError):
            self.sampler.sample(Tensor([], (2, 1000)), None, None, Array([17, 100]),
                                Tensor([4, 0]), None, None, Array([5, 1]), return_logprobs=True)


if __name__ == "__main__":
    unittest.main()
