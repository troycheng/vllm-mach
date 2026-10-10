"""CPU contracts for the 2B mixed GDN port; no torch or GPU is required."""
from __future__ import annotations

import ast
import builtins
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
import types
import unittest
from unittest.mock import patch


GDN = Path(__file__).resolve().parents[1] / "src/vllm_mach/mxfp8/two_b/gdn"
PACKAGE = "mach_2b_mixed_cpu"
NS = types.SimpleNamespace


def _forward_core(self, mixed_qkv, b, a, core_attn_out, hidden_states=None):
    self.events.append("before")
    split_non_spec = self.split
    mixed_qkv_non_spec = mixed_qkv
    ssm_state = self.kv_cache[1]
    non_spec_state_indices_tensor = self.ids
    num_decode_tokens = self.decode_tokens
    if split_non_spec:
        query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec[:num_decode_tokens]
        )
        core_attn_out_decode = (query_decode, key_decode, value_decode)
    else:
        core_attn_out_decode = None
    self.events.append(("after", core_attn_out_decode, ssm_state, hidden_states))
    return "stock-result"


def missing_branch(self, mixed_qkv, b, a, core_attn_out, hidden_states=None):
    return "unchanged"


class FakeTensor:
    def __init__(self, name, rows, start=0):
        self.name, self.shape, self.start = name, (rows,), start

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            return FakeTensor(self.name, len(range(start, stop, step)), self.start + start)
        raise TypeError(item)

    def numel(self):
        return self.shape[0]

    def unsqueeze(self, dim):
        return self

    def transpose(self, dim1, dim2):
        return self

    def stride(self, dim):
        return 1


class FakeKernel:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.events.append((self.name, grid, args, kwargs))
        return launch


def load_mixed():
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.mixed", GDN / "mixed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MixedPortContracts(unittest.TestCase):
    def test_clone_changes_only_mixed_decode_branch_and_rejects_source_drift(self):
        mixed = load_mixed()
        compiled = []
        def capture(source, filename, mode):
            compiled.append(source)
            return builtins.compile(source, filename, mode)
        with patch.dict(mixed.__dict__, {"compile": capture}):
            clone = mixed._clone_core(_forward_core, {"_mach_mixed_decode":
                                                   lambda *args: "mixed-result"})
        original = ast.parse(textwrap.dedent(inspect.getsource(_forward_core))).body[0]
        edited = ast.parse(compiled[0]).body[0]
        changed = [i for i, (left, right) in enumerate(zip(original.body, edited.body,
                                                            strict=True))
                   if ast.dump(left) != ast.dump(right)]
        self.assertEqual(len(changed), 1)
        self.assertEqual(ast.unparse(original.body[changed[0]].test), "split_non_spec")
        self.assertIn("_mach_mixed_decode", ast.unparse(edited.body[changed[0]]))
        layer = NS(events=[], split=False, kv_cache=[None, object()],
                   ids=FakeTensor("ids", 26), decode_tokens=24)
        self.assertEqual(clone(layer, FakeTensor("qkv", 26), None, None,
                               FakeTensor("out", 26), "sentinel"), "stock-result")
        self.assertEqual(layer.events[0], "before")
        self.assertEqual(layer.events[1][0:2], ("after", None))
        with self.assertRaisesRegex(RuntimeError, "Unique stock mixed decode branch absent"):
            mixed._clone_core(missing_branch, {})

    def test_mixed_bounds_fallback_and_prefill_slot_isolation(self):
        events = []
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(GDN)]
        ordered = types.ModuleType(f"{PACKAGE}.ordered_allm_triton")
        stock_math = types.ModuleType(f"{PACKAGE}.mixed_ordered_triton")
        ordered.NV = stock_math.NV = 4
        ordered.HV = stock_math.HV = 16
        for name, value in {"H": 16, "V": 128, "K": 128, "BV": 32}.items():
            setattr(stock_math, name, value)
            setattr(ordered, name, value)
        ordered._ordered_materialize = FakeKernel("materialize", events)
        stock_math._ordered_decode = FakeKernel("decode", events)
        stock_math.validate = lambda base, scratch: (events.append(("validate", base, scratch))
                                                     or (32, 4))
        torch = types.ModuleType("torch")
        torch.cuda = NS(is_current_stream_capturing=lambda: False)

        class Layer:
            def __init__(self):
                self.events = events
                self.prefix = "layer.0"
                self.kv_cache = [None, FakeTensor("base", 32)]
                self.A_log = self.dt_bias = FakeTensor("gate", 16)
                self.split = True
                self.decode_tokens = 24
                self.ids = FakeTensor("ids", 26)
                self.md = None

            def rearrange_mixed_qkv(self, qkv):
                events.append(("stock-decode", qkv.shape[0]))
                return qkv, qkv, qkv

        def published_core(self, *args, **kwargs):
            events.append(("fallback", self.md.num_decodes, self.md.num_prefills))
            return _forward_core(self, *args, **kwargs)

        Layer._forward_core = published_core
        gdn = types.ModuleType("fake_gdn")
        gdn.QwenGatedDeltaNetAttention = Layer
        scratch = NS(**{name: object() for name in
                        ("pending_k", "pending_d", "coeff", "prefix", "age")})
        gw = NS(_READY=False, _VIEW_SCRATCH={}, _ORIG_CORE=_forward_core,
                _LAYERS={"layer.0": object()}, _view_key=lambda base: "one",
                _metadata=lambda layer: layer.md)
        modules = {PACKAGE: package, ordered.__name__: ordered,
                   stock_math.__name__: stock_math, "torch": torch}
        with patch.dict(sys.modules, modules):
            mixed = load_mixed()
            mixed.install(gw, gdn)
            first = Layer._forward_core
            mixed.install(gw, gdn)
            self.assertIs(Layer._forward_core, first)
            gw._VIEW_SCRATCH["one"] = scratch
            gw._READY = True
            layer = Layer()
            qkv, out = FakeTensor("qkv", 200), FakeTensor("out", 200)

            def run(decodes, prefills, *, actual=None, ids=None, spec=0, tokens=None):
                layer.decode_tokens = decodes
                layer.ids = FakeTensor("ids", ids if ids is not None else decodes + prefills)
                layer.md = NS(num_decodes=decodes, num_prefills=prefills,
                              num_decode_tokens=decodes if tokens is None else tokens,
                              num_actual_tokens=actual if actual is not None else decodes + prefills,
                              num_spec_decodes=spec, spec_sequence_masks=None,
                              non_spec_state_indices_tensor=layer.ids)
                events.clear()
                return layer._forward_core(qkv, FakeTensor("b", 200),
                                           FakeTensor("a", 200), out, "sentinel")

            for d in (24, 160):
                self.assertEqual(run(d, 2), "stock-result")
                self.assertEqual([e[0] for e in events if isinstance(e, tuple)],
                                 (["validate", "materialize", "decode", "after"]
                                  if d == 24 else ["materialize", "decode", "after"]))
                materialize = next(e for e in events if isinstance(e, tuple)
                                   and e[0] == "materialize")
                decode = next(e for e in events if isinstance(e, tuple)
                              and e[0] == "decode")
                self.assertEqual(materialize[1], (4, 16, 2))
                self.assertEqual(materialize[2][1].shape[0], 2)
                self.assertEqual(materialize[2][1].start, d)
                self.assertEqual(decode[1], (4, 16, d))
                self.assertEqual(decode[2][7].shape[0], d)
                self.assertEqual(decode[2][7].start, 0)
                self.assertIs(materialize[2][0], decode[2][6])
                self.assertIs(materialize[2][2], decode[2][8])
                self.assertNotIn("stock-decode", [e[0] for e in events if isinstance(e, tuple)])

            for d, p, actual, ids, spec, tokens in ((23, 1, None, None, 0, None),
                                                    (161, 1, None, None, 0, None),
                                                    (0, 2, 2, None, 0, None),
                                                    (24, 0, 24, None, 0, None),
                                                    (24, 1, None, 24, 0, None),
                                                    (24, 1, None, None, 1, None),
                                                    (24, 1, 24, None, 0, None),
                                                    (24, 1, 201, None, 0, None),
                                                    (24, 1, None, None, 0, 25)):
                run(d, p, actual=actual, ids=ids, spec=spec, tokens=tokens)
                self.assertIn(("fallback", d, p), events)
                self.assertFalse(any(e[0] in ("materialize", "decode") for e in events
                                     if isinstance(e, tuple)))
            self.assertEqual(mixed.snapshot()["counts_by_layer"]["layer.0"]["eager_mixed_hit"], 2)


if __name__ == "__main__":
    unittest.main()
