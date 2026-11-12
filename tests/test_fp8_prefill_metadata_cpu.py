"""Actual CPU tensors; hardware stream/copy leaves only are sunk (14 cases)."""
import threading
import importlib.util
import os
from unittest import mock
import pytest




def test_import_is_lightweight_and_missing_dependency_keeps_original_path():
    import builtins
    from pathlib import Path
    source = os.environ.get('MACH_TEST_PREFILL_INPUTS_SOURCE')
    if source is None:
        source = Path(__file__).resolve().parents[1] / 'src/vllm_mach/fp8/_prefill_inputs.py'
    real_import = builtins.__import__

    def no_runtime_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm'):
            raise AssertionError('Heavy dependency during lightweight import: ' + name)
        return real_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location('lightweight_prefill_inputs', source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(builtins, '__import__', no_runtime_import):
        spec.loader.exec_module(module)
    assert not module._METADATA_INITIALIZED
    assert module._METADATA_PARAMETER_TYPES == () and module._BASE_TORCH_FUNCTION is None

    def missing_runtime(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm'):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, '__import__', missing_runtime):
        module._initialize_metadata_parameters()
    assert module._METADATA_INITIALIZED and module._METADATA_PARAMETER_TYPES == ()
    with mock.patch.object(module, '_qualified_metadata_parameters', side_effect=AssertionError('Not lazy-once')):
        module._initialize_metadata_parameters()


def test_readonly_metadata_real_cpu_cases():
    torch = pytest.importorskip('torch')
    pytest.importorskip('vllm')
    from vllm.model_executor import parameter as vp
    torch.set_num_threads(1)
    staged = os.environ.get('MACH_TEST_PREFILL_INPUTS_SOURCE')
    if staged:
        spec = importlib.util.spec_from_file_location('tested_prefill_inputs', staged)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from vllm_mach.fp8 import _prefill_inputs as module
    module._initialize_metadata_parameters()  # This test uses bank.__new__, bypassing __init__.
    assert module._METADATA_PARAMETER_TYPES == (vp.ModelWeightParameter, vp.BlockQuantScaleParameter)
    assert module._BASE_TORCH_FUNCTION is vp.BasevLLMParameter.__torch_function__.__func__
    candidate = module.ActivationBank.prepare
    copies = []
    stream = [71]

    class Stream:
        @property
        def cuda_stream(self):
            return stream[0]

    def bank():
        b = module.ActivationBank.__new__(module.ActivationBank)
        b.device = torch.device('cpu')
        b.stream = b.runtime_stream = 71
        b.thread = b.runtime_thread = threading.get_ident()
        b.phase = 'runtime'
        b.buffers = {n: torch.empty(c, dtype=torch.bfloat16)
                     for n, c in [('core', module.CORE), ('gate', module.GATE),
                                  ('residual', module.RESIDUAL)]}
        b.views = {}
        return b

    def storage(t):
        return str(t.device), t.untyped_storage().data_ptr()

    # CPU tensors retain actual activation layouts; only hardware leaves are sunk.
    b = bank()
    core = torch.empty((2048, 32, 128), dtype=torch.bfloat16)
    gate = torch.empty(module.GATE, dtype=torch.bfloat16).as_strided(
        (2048, 32, 128), (12288, 128, 1))
    residual = torch.empty((2048, 2560), dtype=torch.bfloat16)
    parameters = [cls.__new__(cls, torch.empty((32, 64), dtype=dtype))
                  for cls, dtype in zip((vp.ModelWeightParameter, vp.BlockQuantScaleParameter),
                                        (torch.float8_e4m3fn, torch.float32))]
    args = [core, parameters[0], gate, parameters[1], residual, 3, None]
    readonly = {storage(t) for t in parameters}
    cases = []

    def sink(dst, src, non_blocking=False):
        copies.append((id(dst), id(src), non_blocking))
        return dst

    def compare(name, values, ro, signature=None, expected=None):
        copies[:] = []
        result = candidate(b, values, ro, signature)
        assert result[1] == expected, (name, result[1])
        if expected is None:
            converted, observed = result[0]
            assert [x[1] for x in copies] == [id(core), id(gate), id(residual)], name
            assert all(x[2] for x in copies)
            assert converted[1] is values[1] and converted[3] is values[3]
            for index, role in [(0, 'core'), (2, 'gate'), (4, 'residual')]:
                assert converted[index].untyped_storage().data_ptr() == b.buffers[role].untyped_storage().data_ptr()
                assert converted[index].shape == values[index].shape
                assert converted[index].stride() == values[index].stride()
            for index in (1, 3):
                t = values[index]
                assert observed[index] == ('readonly', str(t.device), t.data_ptr(),
                    tuple(t.shape), tuple(t.stride()), str(t.dtype))
            assert observed[5:] == [('scalar', 'int', 3), ('scalar', 'NoneType', None)]
        else:
            assert result[0] is None and not copies, name
        cases.append(name)
        return result[0][1] if expected is None else None

    with mock.patch.object(torch.cuda, 'current_stream', lambda device: Stream()), \
            mock.patch.object(torch.Tensor, 'copy_', sink):
        signature = compare('baseline-and-three-copy-order', args, readonly)
        compare('signature', args, readonly, signature)
        for name, replacement in [
                ('readonly-storage', parameters[0].clone()),
                ('readonly-shape', parameters[0].view(16, 128)),
                ('readonly-stride', parameters[0].t()),
                ('readonly-dtype', parameters[0].to(torch.float32))]:
            values = list(args); values[1] = replacement
            ro = set(readonly); ro.add(storage(replacement))
            compare(name, values, ro, signature, 'input_signature')
        values = list(args); values[1] = core
        compare('activation-alias-role', values, readonly, expected='activation_layout_role_device')
        compare('lost-readonly-storage', args, set(), expected='activation_layout_role_device')
        values = list(args); values[0] = core.to(torch.float32)
        compare('activation-dtype', values, readonly, expected='activation_layout_role_device')
        values = list(args); values[0] = core.transpose(0, 1)
        compare('activation-shape-stride', values, readonly, expected='activation_layout_role_device')
        stream[0] = 72
        compare('stream-change', args, readonly, expected='stream_mismatch')
        stream[0] = 71
        b.runtime_thread = -1
        compare('thread-change', args, readonly, expected='thread_mismatch')
        b.runtime_thread = threading.get_ident()

        class Foreign(torch.Tensor):
            calls = 0
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return super().__torch_function__(func, types, args, kwargs or {})

        foreign = torch.empty((32, 64)).as_subclass(Foreign)
        values = list(args); values[1] = foreign
        ro = set(readonly); ro.add(storage(foreign))
        Foreign.calls = 0
        candidate(b, values, ro)
        assert Foreign.calls > 0
        compare('foreign-subclass-fallback', values, ro)
        cls = vp.ModelWeightParameter
        original_descriptor = cls.__dict__.get('__torch_function__')
        qualified_hook = vp.BasevLLMParameter.__torch_function__.__func__
        overridden_calls = [0]

        def overridden(subcls, func, types, args=(), kwargs=None):
            overridden_calls[0] += 1
            return qualified_hook(subcls, func, types, args, kwargs or {})

        setattr(cls, '__torch_function__', classmethod(overridden))
        try:
            overridden_calls[0] = 0
            candidate(b, args, readonly)
            assert overridden_calls[0] > 0
            compare('exact-type-hook-override-fallback', args, readonly)
        finally:
            if original_descriptor is None:
                delattr(cls, '__torch_function__')
            else:
                setattr(cls, '__torch_function__', original_descriptor)
    assert not torch.cuda.is_initialized()
    assert len(cases) == 14
