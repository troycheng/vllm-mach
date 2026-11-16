"""Native graph lifecycle and prefix-copy contract without CUDA or numerical claims."""
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import pytest

if not os.environ.get('MACH_TEST_RUNTIME_PROFILE'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from vllm_mach.mxfp6.gdn import prefill_graph as product
from vllm_mach.mxfp6.gdn import _prefill_graph_dispatch as dispatch


def test_prefill_graph_install_selected_once_and_keeps_original_on_skip(monkeypatch):
    calls = []
    worker = NS()
    original, run = object(), object()
    chunk = NS(chunk_gated_delta_rule_fwd=original)
    graph = NS(run=run)
    def enable():
        calls.append('enable')
        chunk.chunk_gated_delta_rule_fwd = run
    graph.enable = enable
    binding = NS(graph=graph, chunk=chunk)
    monkeypatch.setattr(product, '_selected', lambda _: False)
    monkeypatch.setattr(product, 'Binding', lambda _: calls.append('prepare') or binding)
    assert product.install(worker) is False
    assert calls == [] and chunk.chunk_gated_delta_rule_fwd is original
    monkeypatch.setattr(product, '_selected', lambda _: True)
    assert product.install(worker) is True
    assert calls == ['prepare', 'enable']
    assert worker._mach_prefill_graph is binding
    with pytest.raises(RuntimeError, match='already installed'):
        product.install(worker)
    assert calls == ['prepare', 'enable']


def test_prefill_graph_scope_preserves_kv_capacity_and_other_modes(monkeypatch):
    from vllm_mach.mxfp6 import head_scale
    from vllm_mach.mxfp6 import gdn
    monkeypatch.setattr(head_scale, '_selected', lambda _: True)
    legacy = NS(enabled=lambda: True)
    monkeypatch.setattr(gdn, 'legacy', legacy, raising=False)
    config = NS(scheduler_config=NS(max_num_seqs=64),
                cache_config=NS(kv_cache_memory_bytes=8589934592),
                parallel_config=NS(enable_dbo=False), speculative_config=None)
    worker = NS(vllm_config=config)
    assert product._selected(worker)
    for owner, name, changed in ((config.scheduler_config, 'max_num_seqs', 32),
                                 (config.cache_config, 'kv_cache_memory_bytes', 1),
                                 (config.parallel_config, 'enable_dbo', True),
                                 (config, 'speculative_config', object())):
        before = getattr(owner, name)
        setattr(owner, name, changed)
        assert not product._selected(worker)
        setattr(owner, name, before)
    legacy.enabled = lambda: False
    assert not product._selected(worker)
    monkeypatch.setattr(head_scale, '_selected', lambda _: False)
    legacy.enabled = lambda: (_ for _ in ()).throw(AssertionError('unselected import'))
    assert not product._selected(worker)


def _fixture(rows=2035):
    events, fallback = [], []
    class Tensor:
        def __init__(self, shape, dtype='bf16', pointer=None, stride=None):
            self.shape, self.dtype, self.device = tuple(shape), dtype, 'cuda:0'
            self.ndim = len(shape)
            self.pointer = pointer if pointer is not None else 16 * (id(self) + 1)
            acc, strides = 1, []
            for size in reversed(shape):
                strides.append(acc); acc *= size
            self.strides = tuple(reversed(strides)) if stride is None else stride
        def stride(self): return self.strides
        def data_ptr(self): return self.pointer
        def copy_(self, source, non_blocking=False):
            assert self.shape == source.shape and non_blocking
            events.append(('copy', self, source)); return self
        def view(self, *shape): return Tensor(shape, self.dtype, self.pointer)
        def view_as(self, other): return self.view(*other.shape)
        def __getitem__(self, slices):
            shape = list(self.shape)
            for i, part in enumerate(slices):
                start, stop, step = part.indices(shape[i])
                assert start == 0 and step == 1
                shape[i] = stop
            return Tensor(shape, self.dtype, self.pointer, self.strides)
    stream = NS(cuda_stream=123)
    torch = NS(Tensor=Tensor, is_inference_mode_enabled=lambda: True,
               is_grad_enabled=lambda: False,
               cuda=NS(current_stream=lambda _: stream,
                       is_current_stream_capturing=lambda: False),
               empty_like=lambda value: Tensor(value.shape, value.dtype))
    def inputs(nrows):
        return dict(q=Tensor((1,nrows,8,128)), k=Tensor((1,nrows,8,128)),
            v=Tensor((1,nrows,16,128)), g=Tensor((1,nrows,16),'fp32'),
            beta=Tensor((1,nrows,16),'fp32'), initial_state=Tensor((2,16,128,128),'fp16'),
            cu_seqlens=Tensor((3,),'int32'), chunk_indices=Tensor((33,2),'int32'),
            chunk_offsets=Tensor((3,),'int64'), core_attn_out=Tensor((nrows,16,128)))
    obj = dispatch.GraphDispatch.__new__(dispatch.GraphDispatch)
    obj.torch, obj.device, obj.stream = torch, 'cuda:0', stream
    obj.thread, obj.sealed, obj.running = threading.get_ident(), True, False
    obj.original = lambda **kwargs: fallback.append(kwargs) or 'original'
    obj._owners_ok = lambda: True
    stage = inputs(2048)
    outputs = (Tensor((1,2048,16),'fp32'), stage['core_attn_out'].view(1,2048,16,128),
               Tensor((1,2048,16,64),'bf16'), Tensor((2,16,128,128),'fp32'), None, None, None)
    bank = dict(stage=stage, outputs=outputs,
                graph=NS(replay=lambda: events.append(('replay',))),
                specs={name: obj._spec(stage[name]) for name in dispatch.INPUTS+('core_attn_out',)})
    obj.banks = {(2,33): bank}
    args = inputs(rows)
    args.update(scale=128**-.5, output_final_state=True)
    return NS(**locals())


def test_prefill_graph_replay_copies_real_prefix_and_returns_parent_and_fresh_state():
    f = _fixture()
    first = f.obj._run(**f.args)
    assert len(f.events) == 12  # nine inputs, replay, two escaping outputs
    assert all(row[0] == 'copy' for row in f.events[:9])
    assert f.events[9] == ('replay',)
    assert first[0].shape == (1,2035,16)
    assert first[2].shape == (1,2035,16,64)
    assert first[0].data_ptr() == f.outputs[0].data_ptr()
    assert first[2].data_ptr() == f.outputs[2].data_ptr()
    assert first[1].data_ptr() == f.args['core_attn_out'].data_ptr()
    assert first[1].shape == (1,2035,16,128)
    assert first[3].data_ptr() != f.outputs[3].data_ptr()
    second = f.obj._run(**f.args)
    assert second[3].data_ptr() != first[3].data_ptr()
    assert not f.fallback and not f.obj.running


def test_prefill_graph_unsupported_inputs_and_mutation_use_original_once():
    mutations = (
        lambda f: setattr(f.obj, 'sealed', False),
        lambda f: setattr(f.obj, 'thread', -1),
        lambda f: setattr(f.obj.torch.cuda, 'current_stream', lambda _: NS(cuda_stream=999)),
        lambda f: setattr(f.obj.torch.cuda, 'is_current_stream_capturing', lambda: True),
        lambda f: setattr(f.obj, '_owners_ok', lambda: False),
        lambda f: setattr(f.args['g'], 'strides', (1,1,1)),
        lambda f: setattr(f.args['q'], 'dtype', 'fp32'),
        lambda f: f.args.update(scale=1.0),
        lambda f: f.args.update(chunk_indices=f.Tensor((32,2),'int32')),
    )
    for mutate in mutations:
        f = _fixture(); mutate(f)
        assert f.obj._run(**f.args) == 'original'
        assert len(f.fallback) == 1 and f.fallback[0] == f.args and not f.events
    for rows in (0, 2049):
        f = _fixture(rows)
        assert f.obj._run(**f.args) == 'original'
        assert len(f.fallback) == 1 and not f.events
