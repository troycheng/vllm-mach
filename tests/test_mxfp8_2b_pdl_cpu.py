"""PDL policy, arithmetic, and capture routing without a CUDA dependency."""
import ast
import copy
from types import SimpleNamespace as NS
import threading

import pytest

from vllm_mach.mxfp8.two_b import pdl


@pytest.fixture
def state(monkeypatch):
    for name, value in {'_INSTALLED': True, '_READY': True, '_CLONES': {},
                        '_SELECTED': {}, '_COUNTS': pdl.Counter(), '_FULL_MODE': 'FULL',
                        '_DESCRIPTOR_LOCAL': threading.local(), '_CAPTURING': lambda: False}.items():
        monkeypatch.setattr(pdl, name, value)
    return pdl


def bound(state, monkeypatch, rows=16, mode='FULL', warmup=False, capturing=True):
    state._DESCRIPTOR_LOCAL.stack = [dict(rows=rows, mode=mode, warmup=warmup)]
    monkeypatch.setattr(state, '_CAPTURING', lambda: capturing)


class Kernel:
    def __init__(self, config):
        self.launchers = [NS(config=NS(**copy.deepcopy(config)))]
        self.calls = []
        self.answer = object()

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.answer


def prepared_proxy(state, rows=16):
    spec = state.policy()['kernels'][0]
    desc = state._descriptor(spec['name'], spec['original_source'])
    original, clone = Kernel(spec['allowed_configs'][0]), Kernel(spec['allowed_configs'][0])
    proxy = state._Proxy(original, desc)
    index = desc['arg_names'].index('xnumel')
    args = tuple([object()] * index + [rows, 2048])
    state._CLONES[state._kernel_key(spec, rows, spec['allowed_configs'][0])] = clone
    return spec, proxy, original, clone, args


def test_packaged_policy_preserves_arithmetic_and_twenty_config_variants():
    data = pdl.policy()  # Compares complete AST after removing only the PDL controls.
    assert len(data['kernels']) == 7
    assert sum(len(s['allowed_configs']) * len(s['layouts']) for s in data['kernels']) == 20
    assert len({s['semantic_key'] for s in data['kernels']}) == 4
    assert all(len(s['layouts']) == 2 for s in data['kernels'])


def test_source_matching_ignores_comments_but_rejects_changed_math():
    spec = pdl.policy()['kernels'][0]
    assert pdl._descriptor(spec['name'], '# another build path\n' + spec['original_source'])['spec']
    tree = ast.parse(spec['original_source'])
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    changed = next(n for n in ast.walk(fn) if isinstance(n, ast.Constant) and type(n.value) is float)
    changed.value += 0.125
    assert pdl._descriptor(spec['name'], ast.unparse(tree))['spec'] is None


@pytest.mark.parametrize('rows,mode,warmup,capturing,expected', [
    (16,'FULL',False,True,True), (24,'FULL',False,True,True),
    (16,'FULL',True,True,False), (16,'PIECEWISE',False,True,False),
    (16,'FULL',False,False,False), (32,'FULL',False,True,False)])
def test_actual_descriptor_controls_route(state, monkeypatch, rows, mode, warmup, capturing, expected):
    bound(state,monkeypatch,rows,mode,warmup,capturing)
    assert state.native_selected(rows) is expected
    assert not state.native_selected(rows + 1)


@pytest.mark.parametrize('rows', [16,24])
def test_original_warmup_then_clone_preserves_args_and_return(state,monkeypatch,rows):
    spec,proxy,original,clone,args = prepared_proxy(state,rows)
    assert proxy.run(*args,stream=7) is original.answer
    assert len(original.calls)==1 and not clone.calls
    bound(state,monkeypatch,rows)
    assert proxy.run(*args,stream=7) is clone.answer
    assert clone.calls == [(args,{'stream':7})]
    assert len(original.calls)==1
    assert all(a is b for a,b in zip(args,clone.calls[0][0]))
    bound(state,monkeypatch,rows,mode='PIECEWISE')
    assert proxy.run(*args,stream=7) is original.answer
    assert len(original.calls)==2


def test_capture_without_original_warmup_rejected(state,monkeypatch):
    _,proxy,_,_,args = prepared_proxy(state)
    bound(state,monkeypatch)
    with pytest.raises(RuntimeError,match='warmup'):
        proxy.run(*args)


def test_config_drift_rejected_before_clone(state,monkeypatch):
    _,proxy,original,clone,args = prepared_proxy(state)
    proxy.run(*args)
    original.launchers[0].config.num_warps=32
    bound(state,monkeypatch)
    with pytest.raises(RuntimeError,match='configuration'):
        proxy.run(*args)
    assert not clone.calls


def test_unknown_source_falls_back_except_qualified_capture(state,monkeypatch):
    _,proxy,original,_,args = prepared_proxy(state)
    proxy._descriptor['spec']=None
    assert proxy.run(*args) is original.answer
    bound(state,monkeypatch)
    with pytest.raises(RuntimeError,match='source'):
        proxy.run(*args)


def test_descriptor_restored_after_exception(state,monkeypatch):
    class Manager:
        def capture(self,factory):
            return factory(NS(cg_mode='FULL',num_tokens=16),warmup=False)()
    def factory(desc,warmup):
        def forward():
            assert state.native_selected(16)
            raise ValueError('body')
        return forward
    monkeypatch.setattr(state,'_CAPTURING',lambda:True)
    state._bind_manager(Manager)
    with pytest.raises(ValueError,match='body'):
        Manager().capture(factory)
    assert state._current_descriptor() is None


def test_receipt_rejects_missing_producer_or_native_capture(state):
    with pytest.raises(RuntimeError,match='Incomplete'):
        state.verify_capture({16})
    for i,spec in enumerate(state.policy()['kernels']):
        state._SELECTED[str(i)]={'rows':16,'semantic_key':spec['semantic_key']}
    state._COUNTS['producer_capture_m16']=7
    with pytest.raises(RuntimeError,match='Incomplete'):
        state.verify_capture({16})
    state._COUNTS['native_capture_m16']=78
    assert state.verify_capture({16})['ready']
