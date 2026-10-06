"""CPU boundary/scatter tests; these do not qualify NVFP4 GPU arithmetic."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/head.py"
spec = importlib.util.spec_from_file_location("mach_head_cpu_test", PATH)
head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)


class FullLogitsCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch required for CPU tensor/scatter checks")
        cls.torch = torch

    def inputs(self, m=2):
        torch = self.torch
        hidden = types.SimpleNamespace(shape=(m, 2560), ndim=2, dtype=torch.bfloat16,
                                       device=torch.device("cpu"), is_cuda=True,
                                       is_contiguous=lambda: True)
        weight = types.SimpleNamespace(shape=(248320, 2560), dtype=torch.bfloat16,
                                       device=torch.device("cpu"), is_contiguous=lambda: True)
        return hidden, weight

    def test_full_tail_is_retained_and_unsorted_topk_scores_are_replaced(self):
        torch = self.torch
        hidden, weight = self.inputs()
        torch.manual_seed(120)
        original = torch.randn(2, head.VOCAB_SIZE).to(torch.bfloat16)
        events = []
        state = types.SimpleNamespace(candidates=2048, output_size=248320, input_size=2560)
        state.coarse_logits = lambda x, bias: (events.append(("coarse", bias)) or original.clone())
        def refine(x, w, indices, bias):
            events.append(("refine", bias))
            return (indices % 31).to(torch.bfloat16)
        state.refine_logits = refine
        with patch.object(torch, "topk", wraps=torch.topk) as topk:
            output, indices, selected = head.full_logits(hidden, weight, state)
        topk.assert_called_once()
        self.assertEqual(topk.call_args.kwargs, {"dim": 1, "sorted": False})
        self.assertEqual(topk.call_args.args[1], 2048)
        self.assertEqual(output.shape, (2, 248320))
        self.assertEqual(events, [("coarse", None), ("refine", None)])
        self.assertTrue(torch.equal(output.gather(1, indices), selected))
        mask = torch.zeros_like(output, dtype=torch.bool).scatter_(1, indices, True)
        self.assertTrue(torch.equal(output[~mask], original[~mask]))
        expected_indices = torch.topk(original, 2048, dim=1, sorted=False).indices
        self.assertTrue(torch.equal(indices, expected_indices))

    def test_wrong_width_or_refine_result_is_rejected(self):
        torch = self.torch
        hidden, weight = self.inputs()
        state = types.SimpleNamespace(candidates=256, output_size=248320, input_size=2560)
        with self.assertRaises(ValueError):
            head.full_logits(hidden, weight, state)
        state.candidates = 2048
        state.refine_logits = lambda *args: torch.zeros(2, 2048, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            head.full_logits(hidden, weight, state,
                             coarse=torch.zeros(2, 248320, dtype=torch.bfloat16))


class InstallBoundaryCPU(unittest.TestCase):
    def worker_and_dependencies(self):
        class Module:
            def register_parameter(self, name, value):
                setattr(self, name, value)
        torch = types.ModuleType("torch")
        torch.bfloat16 = "bf16"
        torch.nn = types.SimpleNamespace(Module=Module)
        torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False)
        weight = types.SimpleNamespace(is_cuda=True, dtype="bf16", shape=(248320, 2560),
                                       is_contiguous=lambda: True, data_ptr=lambda: 100)
        layer = types.SimpleNamespace(weight=weight)
        original_calls = []
        def original(lm_head, hidden_states, embedding_bias):
            original_calls.append(True)
        processor = types.SimpleNamespace(_apply_head=original, org_vocab_size=248320,
                                          head_dtype=None, soft_cap=None, scale=1.)
        model = types.SimpleNamespace(lm_head=layer, logits_processor=processor,
                    model=types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=weight)))
        sampler = object()
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model, sampler=sampler))
        nv = types.ModuleType("vllm_mach.mxfp6.hybrid_nvfp4_lm_head")
        fp4 = types.SimpleNamespace(candidates=2048, output_size=248320, input_size=2560, backend="b12x",
                                    weight=types.SimpleNamespace(nbytes=head.PACKED_BYTES - 8),
                                    scale=types.SimpleNamespace(nbytes=4), global_scale=types.SimpleNamespace(nbytes=4))
        def prepare(detached, *, candidates):
            self.assertEqual(candidates, 2048)
            self.assertIs(detached.weight, weight)
            detached._hybrid_nvfp4_lm_head_state = fp4
            return True
        nv.prepare_hybrid_nvfp4_lm_head = prepare
        package = types.ModuleType("vllm_mach.mxfp6")
        package.hybrid_nvfp4_lm_head = nv
        return worker, torch, package, original_calls, sampler

    def test_install_once_and_dispatch_to_full_logits_without_sampler_changes(self):
        worker, torch, package, original_calls, sampler = self.worker_and_dependencies()
        with patch.dict(sys.modules, {"torch": torch, "vllm_mach.mxfp6": package}), \
             patch.object(head, "_tensor_sha", return_value=head.HEAD_SHA256), \
             patch.dict(os.environ, {name: "0" for name in
                 ("VLLM_HYBRID_NVFP4_LM_HEAD", "VLLM_HYBRID_MXFP4_LM_HEAD", "VLLM_HYBRID_MXFP8_LM_HEAD")}):
            receipt = head.install_head(worker)
            self.assertTrue(receipt["installed"])
            hidden = types.SimpleNamespace(shape=(32, 2560))
            with patch.object(head, "full_logits", return_value=("complete_logits", None, None)) as run:
                self.assertEqual(worker.model_runner.model.logits_processor._apply_head(
                    worker.model_runner.model.lm_head, hidden, None), "complete_logits")
                run.assert_called_once()
            self.assertEqual(original_calls, [])
            self.assertIs(worker.model_runner.sampler, sampler)
            self.assertEqual(head.inspect_head(worker)["calls_by_rows"], {"32": 1})
            self.assertFalse(head.inspect_head(worker)["private_head_graph"])
            with self.assertRaises(RuntimeError):
                head.install_head(worker)
            torch.cuda.is_current_stream_capturing = lambda: True
            with self.assertRaises(RuntimeError):
                worker.model_runner.model.logits_processor._apply_head(
                    worker.model_runner.model.lm_head, hidden, None)

    def test_weight_identity_mismatch_is_fatal(self):
        worker, torch, package, _, _ = self.worker_and_dependencies()
        with patch.dict(sys.modules, {"torch": torch, "vllm_mach.mxfp6": package}), \
             patch.object(head, "_tensor_sha", return_value="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                head.install_head(worker)
            self.assertEqual(head.inspect_head(worker), {"installed": False})


if __name__ == "__main__":
    unittest.main()
