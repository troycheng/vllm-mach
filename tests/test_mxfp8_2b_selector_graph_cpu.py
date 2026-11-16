"""Selector ownership/fallback lifecycle; the installed test uses real FI dispatch."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SOURCE))
from vllm_mach.mxfp8.two_b import selector_graph as mod


class Value:
    def __init__(self, value=0, shape=(4, 248320), dtype='bf16', device='cuda', contiguous=True):
        self.value, self.shape, self.dtype, self.device, self.contiguous = value, shape, dtype, device, contiguous
    def is_contiguous(self): return self.contiguous
    def copy_(self, other): self.value = other.value
    def clone(self): return Value(self.value, self.shape, self.dtype, self.device)


def fixture():
    graph = object.__new__(mod.SelectorGraph)
    stream = object(); capturing = [False]; current = [stream]
    graph.torch = NS(bfloat16='bf16', is_inference_mode_enabled=lambda: True, cuda=NS(is_current_stream_capturing=lambda: capturing[0], current_stream=lambda device: current[0]))
    graph.state = NS(candidates=128, weight=NS(device='cuda'), candidate_count_for_topk=lambda k: 128 if k <= 32 else 512, identity=1)
    graph.signature_fn = lambda state: state.identity
    graph.signature = 1; graph.stream = stream
    graph.head_graphs = graph.state._head_graphs = object()
    from threading import Lock
    graph.lock = Lock(); graph.replays = graph.fallbacks = 0
    graph.input, graph.indices = Value(), Value(shape=(4, 128))
    events = []
    original_copy = graph.input.copy_
    graph.input.copy_ = lambda x: (events.append('copy'), original_copy(x))[-1]
    def replay(): events.append('replay'); graph.indices.value = graph.input.value
    graph.graph = NS(replay=replay)
    fallback = object()
    graph.original = lambda x, **kwargs: (events.append(('original', x, kwargs)), fallback)[1]
    return NS(graph=graph, capturing=capturing, current=current, events=events, fallback=fallback)


class SelectorCPU(unittest.TestCase):
    def test_current_adjusted_input_two_inflight_and_lock_release(self):
        f = fixture(); a = f.graph.select(Value(7), top_k=20); b = f.graph.select(Value(-9), top_k=20)
        self.assertEqual((a.value, b.value, f.graph.indices.value), (7, -9, -9))
        self.assertIsNot(a, b); self.assertIsNot(b, f.graph.indices)
        self.assertEqual(f.events, ['copy', 'replay', 'copy', 'replay'])
        self.assertEqual((f.graph.replays, f.graph.fallbacks), (2, 0))
        self.assertTrue(f.graph.lock.acquire(False)); f.graph.lock.release()

    def test_unknown_shape_width_dtype_stream_state_capture_and_contention_fallback(self):
        mutations = [lambda f, x: setattr(x, 'shape', (1, 248320)),
                     lambda f, x: setattr(x, 'shape', (4, 256)),
                     lambda f, x: setattr(x, 'dtype', 'fp32'),
                     lambda f, x: setattr(x, 'device', 'other'),
                     lambda f, x: setattr(x, 'contiguous', False),
                     lambda f, x: f.current.__setitem__(0, object()),
                     lambda f, x: setattr(f.graph.state, 'identity', 2),
                     lambda f, x: setattr(f.graph.state, '_head_graphs', None),
                     lambda f, x: setattr(f.graph.torch, 'is_inference_mode_enabled', lambda: False),
                     lambda f, x: f.capturing.__setitem__(0, True),
                     lambda f, x: f.graph.lock.acquire(False)]
        for mutate in mutations:
            f = fixture(); x = Value(); mutate(f, x)
            self.assertIs(f.graph.select(x, top_k=20), f.fallback)
            self.assertEqual(f.events, [('original', x, {'top_k': 20})])
        f = fixture(); x = Value(); self.assertIs(f.graph.select(x, top_k=64), f.fallback)
        self.assertEqual(f.events, [('original', x, {'top_k': 64})])

    def test_replay_exception_releases_lock(self):
        f = fixture(); f.graph.graph.replay = lambda: (_ for _ in ()).throw(RuntimeError('replay'))
        with self.assertRaisesRegex(RuntimeError, 'replay'): f.graph.select(Value(), top_k=20)
        self.assertTrue(f.graph.lock.acquire(False)); f.graph.lock.release()

    def test_selected_runtime_version_guard(self):
        worker = NS(_mach_2b_contract={'mode': 'production', 'ssm_dtype': 'float32', 'hybrid_greedy_head': True})
        with patch.object(mod, 'version', return_value='0.32.0'):
            with self.assertRaisesRegex(RuntimeError, 'selected vLLM runtime'):
                mod.install(worker, NS(_head_graphs=object()))

    def test_unselected_and_head_graph_disabled_preserve_original_without_runtime_import(self):
        with patch.object(mod, 'version', side_effect=AssertionError('runtime import')):
            for contract in ({}, {'mode': 'quality'}, {'mode': 'production', 'ssm_dtype': 'bfloat16'},
                             {'mode': 'production', 'ssm_dtype': 'float32', 'hybrid_greedy_head': False}):
                self.assertFalse(mod.install(NS(_mach_2b_contract=contract), NS()))
            worker = NS(_mach_2b_contract={'mode': 'production', 'ssm_dtype': 'float32', 'hybrid_greedy_head': True})
            self.assertFalse(mod.install(worker, NS(_head_graphs=None)))
        self.assertEqual(mod.receipt(NS()), {'installed': False})


def test_selected_2b_selector_graph_actual_runtime():
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('vllm') is None:
        import pytest
        pytest.skip('requires installed 0.33 runtime')
    import torch
    import flashinfer.topk as fi
    from vllm_mach.mxfp6 import hybrid_nvfp4_lm_head as nv, hybrid_lm_head_utils as hu
    active = {'graph': None}; calls = []
    class Stream:
        cuda_stream = 0
        def synchronize(self): calls.append('startup_sync')
    stream = Stream()
    class Graph:
        def __init__(self): self.ops = []
        def replay(self):
            calls.append('replay')
            for fn in self.ops: fn()
    class Capture:
        def __init__(self, graph): self.graph = graph
        def __enter__(self): active['graph'] = self.graph
        def __exit__(self, *exc): active['graph'] = None
    class Module:
        def cub_topk_workspace_size(self, *args): return 64
        def cub_topk(self, x, ids, vals, workspace, k, tie):
            calls.append(('actual_FI_CUB_frontend', k, tie))
            def run():
                v, i = torch.topk(x, k, sorted=False); vals.copy_(v); ids.copy_(i)
            run()
            if active['graph'] is not None: active['graph'].ops.append(run)
    worker = NS(_mach_2b_contract={'mode': 'production', 'ssm_dtype': 'float32', 'hybrid_greedy_head': True})
    with patch.object(torch.cuda, 'current_stream', lambda *args: stream), \
            patch.object(torch.cuda, 'is_current_stream_capturing', lambda: active['graph'] is not None), \
            patch.object(torch.cuda, 'CUDAGraph', Graph), patch.object(torch.cuda, 'graph', Capture), \
            patch.object(fi, 'get_compute_capability', lambda device: (12, 0)), \
            patch.object(fi, 'get_topk_module', lambda: Module()), \
            patch.object(hu, 'has_flashinfer', lambda: True), \
            patch.object(nv.envs, 'VLLM_HYBRID_NVFP4_LM_HEAD_USE_FLASHINFER_TOPK', True):
        t = torch.zeros(1, dtype=torch.bfloat16)
        state = nv.HybridNvfp4LmHead(t, t, t, 2048, 248320, 128, 64)
        state._head_graphs = NS(stream=stream)
        original = state.select_candidates
        assert not torch.is_inference_mode_enabled()
        assert mod.install(worker, state) and not mod.install(worker, state)
        assert getattr(state, mod._STATE).input.is_inference()
        with torch.inference_mode():
            assert state.select_candidates.__func__ is not nv.HybridNvfp4LmHead.select_candidates
            a = torch.arange(4 * 248320).reshape(4, 248320).remainder(113).to(torch.bfloat16)
            rng = torch.get_rng_state().clone(); first = state.select_candidates(a, top_k=20); saved = first.clone()
            second = state.select_candidates(-a, top_k=20)
            assert torch.equal(first, saved) and first.data_ptr() != second.data_ptr()
            assert first.data_ptr() != getattr(state, mod._STATE).indices.data_ptr()
            assert torch.equal(first, original(a, top_k=20))
            assert torch.equal(second, original(-a, top_k=20))
            assert torch.equal(state.select_candidates(a[:1], top_k=20), original(a[:1], top_k=20))
            assert torch.equal(state.select_candidates(a, top_k=64), original(a, top_k=64))
            assert torch.equal(torch.get_rng_state(), rng)
            receipt = mod.receipt(state); assert receipt['replays'] == 2 and receipt['fallbacks'] == 2
            assert receipt['input_bytes'] == 1986560 and len(receipt['source_sha256']) == 64
            json.dumps(receipt)
        # No global class override: a second sidecar keeps its class method.
        untouched = nv.HybridNvfp4LmHead(t, t, t, 2048, 248320, 128, 64)
        assert 'select_candidates' not in vars(untouched)
        assert calls.count('startup_sync') == 1
        assert any(isinstance(c, tuple) and c[0] == 'actual_FI_CUB_frontend' and c[1] == 128 for c in calls)
        assert not torch.cuda.is_initialized()
