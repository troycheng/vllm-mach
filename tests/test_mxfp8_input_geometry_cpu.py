"""Immutable logit geometry: source boundary and real 0.33 InputBatch parity."""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_mach.mxfp8 import input_geometry as mod


def example(self, value: int = 3, *, label="original") -> tuple:
    draft_tokens = {}
    num_reqs = value
    idx_mapping = label
    if not draft_tokens and self.model_state.num_new_sampled_tokens_per_step == 1:
        return SHARED_ALIAS
    else:
        return value, label
    return expanded_idx_mapping


class GeometryCPU(unittest.TestCase):
    def test_live_globals_defaults_and_only_one_body_changed(self):
        source = textwrap.dedent(inspect.getsource(example))
        with patch.object(mod, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest()):
            selected = mod._transform(example)
        self.assertIs(selected.__globals__, example.__globals__)
        self.assertIs(selected.__defaults__, example.__defaults__)
        self.assertIs(selected.__kwdefaults__, example.__kwdefaults__)
        self.assertIs(selected.__annotations__, example.__annotations__)
        # Original aliases remain live; the general branch is untouched.
        self.assertEqual(selected(NS(model_state=NS(num_new_sampled_tokens_per_step=2))), (3, "original"))
        self.assertIsNone(mod._transform(example))
        tree = ast.parse(source)
        target = next(n for n in ast.walk(tree) if isinstance(n, ast.If))
        saved = target.body
        target.body = ast.parse(mod._REPLACEMENT).body
        target.body = saved
        self.assertEqual(ast.dump(tree), ast.dump(ast.parse(source)))

    def test_only_2b_profile_installs_shared_geometry(self):
        root = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8"
        def calls(path):
            tree = ast.parse(path.read_text())
            return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "input_geometry.install"]
        self.assertEqual(len(calls(root / "two_b/profile.py")), 1)
        self.assertEqual(calls(root / "profile.py"), [])

    def test_unsupported_cold_context_keeps_original(self):
        for name, value in [("speculative_config", object()), ("enable_dbo", True),
                            ("prefill_context_parallel_size", 2), ("decode_context_parallel_size", 2),
                            ("kv_sharing_fast_prefill", True), ("tensor_parallel_size", 2)]:
            runner = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})()
            runner.prepare_inputs = example.__get__(runner)
            cfg = NS(speculative_config=None, parallel_config=NS(tensor_parallel_size=1,
                pipeline_parallel_size=1, prefill_context_parallel_size=1,
                decode_context_parallel_size=1, enable_dbo=False), cache_config=NS(kv_sharing_fast_prefill=False))
            obj = cfg if name == "speculative_config" else cfg.cache_config if name == "kv_sharing_fast_prefill" else cfg.parallel_config
            setattr(obj, name, value)
            before = runner.prepare_inputs
            mod.install(NS(model_runner=runner, vllm_config=cfg))
            self.assertIs(runner.prepare_inputs, before)
            self.assertFalse(hasattr(runner, "_mach_simple_logit_geometry"))

    @unittest.skipUnless(importlib.util.find_spec("vllm"), "requires installed vLLM 0.33")
    def test_actual_prepare_inputs_all_fields_and_retained_prefixes(self):
        actual_cpu()


def actual_cpu():
    import dataclasses
    import builtins
    import copy
    import numpy as np
    import torch
    import vllm.v1.worker.gpu.model_runner as mr
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.config import ParallelConfig, CacheConfig

    original_tree = ast.parse(textwrap.dedent(inspect.getsource(mr.GPUModelRunner.prepare_inputs)))
    captured = []
    def compile_actual(tree, *args, **kwargs):
        captured.append(copy.deepcopy(tree))
        return builtins.compile(tree, *args, **kwargs)
    with patch.object(mod, "compile", compile_actual, create=True):
        transformed = mod._transform(mr.GPUModelRunner.prepare_inputs)
    assert transformed is not None and len(captured) == 1
    restored = captured[0]
    original_branch = next(n for n in ast.walk(original_tree) if isinstance(n, ast.If) and ast.unparse(n.test) == mod._BRANCH)
    target_branch = next(n for n in ast.walk(restored) if isinstance(n, ast.If) and ast.unparse(n.test) == mod._BRANCH)
    assert ast.dump(ast.Module(body=target_branch.body, type_ignores=[])) == ast.dump(ast.parse(mod._REPLACEMENT))
    target_branch.body = original_branch.body
    assert ast.dump(restored) == ast.dump(original_tree)

    def h2d(array, device=None, dtype=None, out=None):
        t = torch.tensor(array, dtype=dtype)
        if out is not None:
            out.copy_(t)
            return out
        return t
    def positions(idx, query, computed, out, seq):
        n = idx.numel()
        out[:n] = computed[idx]
        seq[:n] = computed[idx] + 1
    for maximum in (128, 160):
        for ssm in ("float32", "bfloat16"):
            runner = object.__new__(mr.GPUModelRunner)
            runner.max_num_reqs, runner.max_num_tokens, runner.device = maximum, 2048, torch.device("cpu")
            cfg = NS(speculative_config=None, parallel_config=ParallelConfig(),
                cache_config=CacheConfig(mamba_ssm_cache_dtype=ssm))
            runner.input_buffers = InputBuffers(maximum, 2048, runner.device)
            runner.model_config = NS(rswa_window=None)
            runner.model_state = NS(num_new_sampled_tokens_per_step=1)
            runner.adaptive_verification = runner.fast_prefill = runner.pcp_manager = None
            runner.req_states = NS(num_computed_tokens=NS(gpu=torch.full((maximum,), 100, dtype=torch.int32)),
                num_computed_tokens_np=np.full(maximum, 100, dtype=np.int32), prefill_len=NS(gpu=torch.full((maximum,), 3000, dtype=torch.int32)),
                last_sampled_tokens=torch.zeros(maximum, dtype=torch.int32), draft_tokens=torch.zeros((maximum, 1), dtype=torch.int32))
            original = runner.prepare_inputs
            worker = NS(model_runner=runner, vllm_config=cfg)
            mod.install(worker)
            assert mod.inspect_worker(worker)["installed"]
            selected = runner.prepare_inputs
            mod.install(worker)
            assert runner.prepare_inputs is selected
            assert selected.__func__.__globals__ is original.__func__.__globals__
            retained = []
            with patch.object(mr, "async_tensor_h2d", h2d), patch.object(mr, "prepare_pos_seq_lens", positions), \
                    patch.object(mr, "combine_sampled_and_draft_tokens", lambda *args: torch.arange(int(args[8]), dtype=torch.int64)), torch.inference_mode():
                for rows in (4, 24, 32, 64, maximum, 24):
                    batch = NS(num_tokens=rows, req_ids=[str(i) for i in range(rows)], num_scheduled_tokens=np.ones(rows, dtype=np.int32),
                        idx_mapping_np=np.arange(rows - 1, -1, -1, dtype=np.int32), num_draft_tokens_np=None,
                        has_prefill=False, prefill_len_np=np.full(rows, 3000, dtype=np.int32), num_computed_prefill_tokens_np=np.zeros(rows, dtype=np.int32),
                        is_prefilling_np=np.zeros(rows, dtype=bool), prefill_runs_as_decode_np=np.zeros(rows, dtype=bool),
                        decode_graph_eligible=True, max_seq_len_np=None)
                    args = (NS(scheduled_spec_decode_tokens={}, has_structured_output_requests=False), batch, NS(num_tokens=rows, num_reqs=rows), 0)
                    expected, actual = original(*args), selected(*args)
                    for f in dataclasses.fields(expected):
                        a, b = getattr(expected, f.name), getattr(actual, f.name)
                        if isinstance(a, torch.Tensor): assert torch.equal(a, b), f.name
                        elif isinstance(a, np.ndarray): assert np.array_equal(a, b), f.name
                        else: assert a == b, f.name
                    assert actual.expanded_idx_mapping is actual.idx_mapping
                    retained.append((actual.cu_num_logits, actual.cu_num_logits.clone(), actual.expanded_local_pos, actual.expanded_local_pos.clone()))
                    assert all(torch.equal(a, b) and torch.equal(c, d) for a, b, c, d in retained[-2:])
            assert len(runner._mach_simple_logit_geometry) == maximum + 1
            for rows, (cpu, gpu, zero) in enumerate(runner._mach_simple_logit_geometry):
                assert np.array_equal(cpu, np.arange(rows + 1, dtype=np.int32)) and torch.equal(gpu, torch.arange(rows + 1, dtype=torch.int32))
                assert torch.equal(zero, torch.zeros(rows, dtype=torch.int32))
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    unittest.main()
