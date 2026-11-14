"""Qualified presence arithmetic, instance ownership, finite dispatch and startup."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SOURCE))
from vllm_mach.mxfp8.two_b import presence_graph as mod


def canonical_ast(node):
    # Python 3.12 adds this empty schema field to function definitions.
    # Keep nonempty type parameters significant; normalize only absent vs [].
    node = copy.deepcopy(node)
    for part in ast.walk(node):
        if getattr(part, "type_params", None) == []:
            delattr(part, "type_params")
    return ast.dump(node, include_attributes=False)


class Value:
    def __init__(self, value=0, shape=(4, 2048), dtype='bf16', device='cuda', contiguous=True):
        self.value, self.shape, self.dtype, self.device, self.contiguous = value, shape, dtype, device, contiguous
    @property
    def ndim(self): return len(self.shape)
    def stride(self): return (self.shape[1], 1) if self.ndim == 2 else (1,)
    def data_ptr(self): return id(self)
    def is_contiguous(self): return self.contiguous
    def copy_(self, other): self.value = other.value
    def clone(self): return Value(self.value, self.shape, self.dtype, self.device)


def fixture():
    g = object.__new__(mod.PresenceHeadGraph)
    stream = object(); current = [stream]; capturing = [False]
    g.torch = N(bfloat16='bf16', is_inference_mode_enabled=lambda: True,
                cuda=N(is_current_stream_capturing=lambda: capturing[0], current_stream=lambda device: current[0]))
    g.head = N(weight=Value(shape=(248320, 2048)), shard_indices=N(num_elements_padded=248320))
    g.weight = g.head.weight; g.weight_identity = mod.identity(g.weight)
    g.shard = g.head.shard_indices; g.shard_fields = mod.shard_identity(g.shard)
    g.state = N(input_size=2048, output_size=248320, candidates=128, identity=1, _head_graphs=object())
    g.signature_fn = lambda state: state.identity; g.signature = 1; g.head_graphs = g.state._head_graphs
    g.proxy = N(_get_hybrid_lm_head_state=lambda head: g.state, hybrid_lm_head_enabled=True,
                hybrid_lm_head_row_mask=None, soft_cap=None, scale=1)
    g.counts = Value(3, shape=(160, 248320), dtype='int32'); g.counts_identity = mod.identity(g.counts)
    h, p, s = Value(5), Value(2, shape=(4,), dtype='float32'), Value(1, shape=(4,), dtype='int64')
    g.hidden, g.presence, g.slots = h.clone(), p.clone(), s.clone()
    g.inputs = [(tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in (h, p, s)]
    g.top_k, g.temperature, g.stream, g.lock = 20, mod._TEMPERATURE, stream, Lock()
    g.outputs = (Value(shape=(4, 20), dtype='float32'), Value(shape=(4, 20), dtype='int64'), 0, None)
    events = []
    def replay():
        events.append('replay')
        for out in g.outputs[:2]: out.value = g.hidden.value + g.presence.value + g.slots.value + g.counts.value
    g.graph = N(replay=replay)
    fallback = object()
    g.original = lambda *a, **kw: (events.append(('original', a, kw)), fallback)[1]
    kw = dict(top_k=20, temperature=mod._TEMPERATURE, presence_penalties=p,
              output_token_counts=g.counts, presence_request_indices=s)
    return N(g=g, h=h, p=p, s=s, kw=kw, current=current, capturing=capturing, events=events, fallback=fallback)


class PresenceCPU(unittest.TestCase):
    def test_live_history_two_output_owners_and_no_hot_counter(self):
        f = fixture(); g = f.g
        first = g.get(g.head, f.h, **f.kw)
        self.assertEqual(first[0].value, 11)
        f.h.value = 7; f.p.value = -2; f.s.value = 0; g.counts.value = 9
        second = g.get(g.head, f.h, **f.kw)
        self.assertEqual((first[0].value, second[0].value), (11, 14))
        self.assertIsNot(first[0], second[0]); self.assertIsNot(first[0], g.outputs[0])
        self.assertEqual(g.counts.value, 9)
        self.assertEqual(f.events, ['replay', 'replay'])
        self.assertFalse(hasattr(g, 'replays')); self.assertFalse(hasattr(g, 'trace'))
        self.assertTrue(g.lock.acquire(False)); g.lock.release()

    def test_finite_shape_state_stream_and_full_signature_fallback(self):
        changes = [lambda f: setattr(f.h, 'shape', (1, 2048)), lambda f: setattr(f.h, 'dtype', 'fp32'),
                   lambda f: f.current.__setitem__(0, object()), lambda f: f.capturing.__setitem__(0, True),
                   lambda f: setattr(f.g.state, 'identity', 2), lambda f: setattr(f.g.head, 'weight', Value()),
                   lambda f: f.kw.update(temperature=.8), lambda f: f.kw.update(top_k=64),
                   lambda f: f.kw.update(presence_penalties=None), lambda f: f.kw.update(output_token_counts=Value()),
                   lambda f: f.kw.update(output_token_ids=Value()), lambda f: f.kw.update(embedding_bias=Value()),
                   lambda f: f.kw.update(output_unique_token_ids=Value()), lambda f: f.kw.update(num_output_unique_tokens=Value()),
                   lambda f: f.kw.update(_hybrid_enabled=True), lambda f: f.g.lock.acquire(False)]
        for change in changes:
            f = fixture(); change(f)
            self.assertIs(f.g.get(f.g.head, f.h, **f.kw), f.fallback)
            self.assertEqual(len(f.events), 1)
            self.assertEqual(f.events[0][0], 'original')

    def test_replay_exception_releases_lock(self):
        f = fixture(); f.g.graph.replay = lambda: (_ for _ in ()).throw(RuntimeError('replay'))
        with self.assertRaisesRegex(RuntimeError, 'replay'): f.g.get(f.g.head, f.h, **f.kw)
        self.assertTrue(f.g.lock.acquire(False)); f.g.lock.release()

    def test_unselected_contract_and_headgraph_off_do_not_initialize_proxy(self):
        contracts = [{}, {'mode':'quality'}, {'mode':'production','ssm_dtype':'bfloat16'},
                     {'mode':'production','ssm_dtype':'float32','hybrid_greedy_head':False}]
        with patch('importlib.metadata.version', side_effect=AssertionError('runtime import')):
            for contract in contracts:
                self.assertFalse(mod.install(N(_mach_2b_contract=contract), N()))
            c = {'mode':'production','ssm_dtype':'float32','hybrid_greedy_head':True}
            self.assertFalse(mod.install(N(_mach_2b_contract=c), N(_head_graphs=None)))
        self.assertEqual(mod.receipt(N()), {'installed':False})

    def test_canonical_normalizes_only_empty_schema_field(self):
        source = ast.parse("def example(x):\n    return x * x\n")
        absent = copy.deepcopy(source)
        for node in ast.walk(absent):
            if hasattr(node, 'type_params'): delattr(node, 'type_params')
        self.assertEqual(canonical_ast(source), canonical_ast(absent))
        changed = copy.deepcopy(source)
        # Simulate the new schema when this test runs on an older parser.
        if 'type_params' not in changed.body[0]._fields:
            changed.body[0]._fields = (*changed.body[0]._fields, 'type_params')
        changed.body[0].type_params = [ast.Name(id='T', ctx=ast.Load())]
        self.assertNotEqual(canonical_ast(changed), canonical_ast(source))
        arithmetic = ast.parse("def example(x):\n    return x + x\n")
        self.assertNotEqual(canonical_ast(arithmetic), canonical_ast(source))

    def test_qualified_body_and_original_method_signature(self):
        tree = ast.parse(inspect.getsource(mod.PresenceHeadGraph))
        get = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'get')
        digest = hashlib.sha256(canonical_ast(get).encode()).hexdigest()
        self.assertEqual(digest, '166dc937b719ffa9602691acaa51ad9ae0a61839034b3fe8b6843f9ec8fa37d5')
        body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'body')
        self.assertEqual(hashlib.sha256(canonical_ast(body).encode()).hexdigest(), 'af41e73882aa3543aa3b47fe24672a3dc4d09da1197b7596451912480adfb80d')
        module = ast.parse((SOURCE/'vllm_mach/mxfp6/sampling.py').read_text())
        original = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == '_get_local_topk')
        for args in (get.args, original.args):
            for node in ast.walk(args):
                if isinstance(node, ast.arg): node.annotation = None
        self.assertEqual(ast.dump(get.args), ast.dump(original.args))
        # Startup writes are identical to try_sample's original cold setup.
        setup = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == 'try_sample')
        product = ast.parse(inspect.getsource(mod.install))
        for attr in ('_mach_sampling_proxy','_mach_sampling_head','_mach_sampling_tied_embedding','_mach_sampling_stats'):
            def assignment(t):
                return next(n for n in ast.walk(t) if isinstance(n, ast.Assign) and any(isinstance(x, ast.Attribute)and x.attr==attr for x in n.targets))
            self.assertEqual(ast.dump(assignment(setup)), ast.dump(assignment(product)), attr)
        lifecycle = ast.parse((SOURCE/'vllm_mach/mxfp8/two_b/hybrid_head.py').read_text())
        install = next(n for n in lifecycle.body if isinstance(n, ast.FunctionDef)and n.name=='install')
        calls = [ast.unparse(n.func) for n in ast.walk(install) if isinstance(n, ast.Call)]
        self.assertLess(calls.index('nv.capture_hybrid_nvfp4_lm_head'), calls.index('presence_graph.install'))
        self.assertLess(calls.index('selector_graph.install'), calls.index('presence_graph.install'))


def test_selected_2b_presence_graph_actual_runtime():
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('vllm') is None:
        import pytest
        pytest.skip('requires installed 0.33 runtime')
    import torch
    import numpy as np
    from vllm_mach.mxfp6.sampling import SamplingHead
    from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import HybridNvfp4LmHead
    t = torch.zeros(1, dtype=torch.bfloat16)
    state = HybridNvfp4LmHead(t, t, t, 2048, 248320, 128, 64)
    state._head_graphs = N(stream=object())
    head = N(weight=t, shard_indices=N(num_elements_padded=248320))
    processor = N(head_dtype=torch.bfloat16, soft_cap=None, scale=1)
    embedding = N(weight=t)
    model = N(lm_head=head, logits_processor=processor, model=N(embed_tokens=embedding))
    counts = torch.zeros((4, 248320), dtype=torch.int32)
    runner = N(model=model, sampler=N(penalties_state=N(output_bin_counts=counts)))
    worker = N(model_runner=runner, _mach_2b_contract={'mode':'production','ssm_dtype':'float32','hybrid_greedy_head':True})
    records = []
    class Graph:
        def __init__(self, proxy, h, hidden, presence, pool, slots, *, top_k, temperature):
            assert type(proxy)is SamplingHead and proxy.state is state and pool is counts and h is head
            assert hidden.shape==(4,2048)and presence.shape==slots.shape==(4,)
            assert hidden.dtype==torch.bfloat16 and presence.dtype==torch.float32 and slots.dtype==torch.int64
            assert hidden.is_inference()and slots.is_inference()and top_k==20 and temperature==mod._TEMPERATURE
            self.original = proxy._get_local_topk
            records.append((proxy,pool))
        def get(self,*args,**kwargs): return self.original(*args,**kwargs)
    rng = torch.get_rng_state().clone(); nrng = np.random.get_state()
    with patch.object(mod,'PresenceHeadGraph',Graph), patch.object(torch.cuda,'is_current_stream_capturing',lambda:False):
        assert mod.install(worker,state)and not mod.install(worker,state)
    proxy = runner._mach_sampling_proxy
    assert len(records)==1 and records[0][0]is proxy and runner._mach_sampling_head==(head,t.data_ptr())
    assert runner._mach_sampling_tied_embedding is embedding and runner._mach_sampling_stats=={'topk_calls':0,'full_calls':0,'rows':0}
    assert proxy._get_local_topk.__self__ is getattr(worker,mod._STATE)and getattr(worker,mod._STATE).original.__func__ is SamplingHead._get_local_topk
    assert proxy._compact_topk_sample_seeds is proxy._full_vocab_sample_seeds is None
    assert torch.equal(rng,torch.get_rng_state())and all(np.array_equal(a,b)for a,b in zip(nrng,np.random.get_state()))
    small_runner = N(model=model, sampler=N(penalties_state=N(output_bin_counts=counts[:3])))
    small_worker = N(model_runner=small_runner, _mach_2b_contract=worker._mach_2b_contract)
    with patch.object(mod, 'PresenceHeadGraph', side_effect=AssertionError('small pool must not capture')), \
            patch.object(torch.cuda, 'is_current_stream_capturing', lambda: False):
        assert not mod.install(small_worker, state)
    assert not hasattr(small_runner, '_mach_sampling_proxy')
    assert not torch.cuda.is_initialized()


if __name__=='__main__':unittest.main()
