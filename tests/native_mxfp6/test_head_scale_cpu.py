"""Native head owner and installation behavior; no GPU or numerical re-proof."""
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

if not os.environ.get('MACH_TEST_RUNTIME_PROFILE'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from vllm_mach.mxfp6 import head_scale as mod
from vllm_mach.mxfp8 import two_b


def fixture():
    device = NS(index=0)
    flags = dict(current=0, capture=False)
    launches, outputs, coarse_calls = [], [], []

    class Tensor:
        is_cuda = True
        dtype = 'bf16'
        def __init__(self, shape=(16, 2048), value=4.0):
            self.shape, self.value, self.device = shape, value, device
        def stride(self): return (2048, 1) if len(self.shape) == 2 else ()
        def data_ptr(self): return id(self)
        def is_conj(self): return False
        def is_neg(self): return False
        def __rmul__(self, other): return other * self.value
        def __mul__(self, other): return self.value * other.value

    def empty(*args, **kwargs):
        output = Tensor(())
        outputs.append((output, args, kwargs))
        return output

    torch = NS(Tensor=Tensor, bfloat16='bf16', float32='fp32', empty=empty,
        reciprocal=lambda value: 1.0 / value,
        is_inference_mode_enabled=lambda: True, is_grad_enabled=lambda: False,
        compiler=NS(is_compiling=lambda: False),
        jit=NS(is_tracing=lambda: False, is_scripting=lambda: False),
        cuda=NS(current_device=lambda: flags['current'],
                is_current_stream_capturing=lambda: flags['capture']))
    native = types.ModuleType('native_head_fixture')
    native.__file__ = str(Path(mod.__file__).with_name('hybrid_nvfp4_lm_head.py'))
    code = compile(Path(native.__file__).read_bytes(), native.__file__, 'exec', dont_inherit=True)
    scale_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == '_global_scale')
    class_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'HybridNvfp4LmHead')
    coarse_code = next(c for c in class_code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'coarse_logits')
    native._global_scale = types.FunctionType(scale_code, vars(native))
    coarse = types.FunctionType(coarse_code, vars(native))
    native.HybridNvfp4LmHead = type('HybridNvfp4LmHead', (), {'coarse_logits': coarse})
    native._NVFP4_MAX_VALUE, native._BLOCK_SIZE, native.torch = 2688.0, 16, torch
    getter_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'get_hybrid_nvfp4_lm_head')
    native._STATE_NAME = '_hybrid_nvfp4_lm_head_state'
    native.get_hybrid_nvfp4_lm_head = types.FunctionType(getter_code, vars(native))
    native.flashinfer_nvfp4_quantize_128x4 = lambda hidden, scale: (hidden, scale)

    class Logits:
        def __getitem__(self, item): return self
    logits = Logits()
    def mm(*args, **kwargs):
        coarse_calls.append((args, kwargs)); return logits
    native.flashinfer_scaled_fp4_mm = mm

    kernel = types.ModuleType('kernel_fixture')
    kernel.__file__ = str(Path(two_b.__file__).with_name('_head_scale_kernel.py'))
    kernel_code = compile(Path(kernel.__file__).read_bytes(), kernel.__file__, 'exec', dont_inherit=True)
    class JIT:
        def __init__(self, fn): self.fn = fn
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, args, kwargs))
    for name in ('reduce_scale', 'scalar_finalize'):
        function_code = next(c for c in kernel_code.co_consts if isinstance(c, types.CodeType) and c.co_name == name)
        setattr(kernel, name, JIT(types.FunctionType(function_code, vars(kernel))))

    def state():
        value = native.HybridNvfp4LmHead()
        value.input_size, value.output_size = 2048, 124160
        value.candidates, value.max_rows, value.backend = 128, 0, 'cudnn'
        value._head_graphs = None
        value.weight, value.scale, value.global_scale = Tensor((124160, 2048)), Tensor(()), Tensor(())
        return value
    attached, proxy_state, unrelated = state(), state(), state()
    head = NS(weight=attached.weight, tp_size=2)
    setattr(head, native._STATE_NAME, attached)

    class Processor:
        def get_top_tokens(self, *args): pass
    class Qwen3_5Model: pass
    class Qwen3_5MoeForCausalLM:
        def get_top_tokens(self, *args): pass
    class Qwen3_5MoeForConditionalGeneration:
        def get_top_tokens(self, *args): pass
    qwen = NS(Qwen3_5Model=Qwen3_5Model, Qwen3_5MoeForCausalLM=Qwen3_5MoeForCausalLM,
              Qwen3_5MoeForConditionalGeneration=Qwen3_5MoeForConditionalGeneration)
    backbone = Qwen3_5Model(); backbone.layers = [object(), object()]
    language = Qwen3_5MoeForCausalLM()
    language.model, language.lm_head, language.logits_processor = backbone, head, Processor()
    outer = Qwen3_5MoeForConditionalGeneration(); outer.language_model = language
    proxy = NS(processor=language.logits_processor, _get_hybrid_lm_head_state=lambda _: proxy_state)
    runner = NS(model=outer, _mach_sampling_proxy=proxy, _mach_sampling_head=(head, head.weight.data_ptr()),
                cudagraph_manager=NS(_graphs_captured=True, graphs={16: object()}))
    config = NS(model_config=NS(enforce_eager=False,
                    hf_text_config=NS(model_type='qwen3_5_moe_text', hidden_size=2048)),
                cache_config=NS(mamba_ssm_cache_dtype='float16'),
                parallel_config=NS(tensor_parallel_size=2),
                compilation_config=NS(mode='compile', cudagraph_mode='full'))
    worker = NS(vllm_config=config, use_v2_model_runner=True, model_runner=runner)
    modules = {'torch.utils._python_dispatch': NS(_get_current_dispatch_mode=lambda: None),
               'torch.overrides': NS(_get_current_function_mode=lambda: None),
               'vllm.model_executor.models': NS(qwen3_5=qwen),
               'vllm.config': NS(CompilationMode=NS(VLLM_COMPILE='compile'),
                                 CUDAGraphMode=NS(FULL_AND_PIECEWISE='full'))}
    return NS(**locals())


def test_native_head_install_is_state_local_and_detaches():
    f = fixture(); original = f.native.HybridNvfp4LmHead.coarse_logits
    with patch.dict(sys.modules, f.modules), patch.object(mod, '_runtime', return_value=(f.torch, f.native)), \
         patch.object(two_b, '_head_scale_kernel', f.kernel, create=True):
        assert mod.install(f.worker)
        binding = getattr(f.worker, mod._STATE)
        assert mod.inspect_worker(f.worker)['installed']
        assert not f.launches and not f.coarse_calls
        assert f.native._global_scale is binding.reference
        assert f.native.HybridNvfp4LmHead.coarse_logits is original
        assert f.unrelated.coarse_logits.__func__ is original
        assert f.attached.coarse_logits.__func__ is f.proxy_state.coarse_logits.__func__
        assert binding.clone.__code__ is original.__code__
        assert binding.clone.__defaults__ is original.__defaults__
        assert binding.clone.__kwdefaults__ is original.__kwdefaults__
        assert all(binding.clone.__globals__[key] is value for key, value in vars(f.native).items()
                   if key != '_global_scale')
        hidden = f.Tensor()
        assert f.attached.coarse_logits(hidden, None) is f.logits
        assert f.proxy_state.coarse_logits(hidden, None) is f.logits
        assert len(f.launches) == 2
        assert f.outputs[0][0] is not f.outputs[1][0]
        assert f.launches[0] == ((1,), (hidden, f.outputs[0][0]), dict(num_warps=8, enable_fp_fusion=False))
        assert f.coarse_calls[0][0] == (hidden, f.attached.weight, f.outputs[0][0], f.attached.scale)
        with pytest.raises(RuntimeError, match='already installed'): mod.install(f.worker)
        binding.detach()
        assert not mod.inspect_worker(f.worker)['installed']
        assert all('coarse_logits' not in vars(state) for state in (f.attached, f.proxy_state, f.unrelated))
        assert f.native._global_scale is binding.reference


def test_native_head_rejects_foreign_owners_without_overwriting():
    for mutation, match in (
        (lambda f: setattr(f.runner, 'model', NS(language_model=f.language)), 'outer model'),
        (lambda f: setattr(f.runner._mach_sampling_proxy, 'processor', object()), 'another head'),
        (lambda f: setattr(f.native._global_scale, '__code__', (lambda x: x).__code__), 'function owner'),
    ):
        f = fixture(); mutation(f)
        with patch.dict(sys.modules, f.modules), pytest.raises(RuntimeError, match=match):
            mod.Binding(f.worker, f.torch, f.native, f.kernel)
        assert 'coarse_logits' not in vars(f.attached)
    f = fixture()
    with patch.dict(sys.modules, f.modules):
        binding = mod.Binding(f.worker, f.torch, f.native, f.kernel)
        binding.on()
        foreign = types.MethodType(lambda self, *args: None, f.attached)
        f.attached.coarse_logits = foreign
        with pytest.raises(RuntimeError, match='owner changed'): binding.detach()
        assert f.attached.coarse_logits is foreign


def test_native_head_scope_skips_without_runtime_import():
    mutations = [
        lambda f: setattr(f.worker, 'use_v2_model_runner', False),
        lambda f: setattr(f.config.model_config, 'enforce_eager', True),
        lambda f: setattr(f.config.cache_config, 'mamba_ssm_cache_dtype', 'float32'),
        lambda f: setattr(f.config.model_config.hf_text_config, 'model_type', 'qwen3_5_text'),
        lambda f: setattr(f.config.model_config.hf_text_config, 'hidden_size', 4096),
        lambda f: setattr(f.config.parallel_config, 'tensor_parallel_size', 1),
        lambda f: setattr(f.config.compilation_config, 'mode', 'none'),
        lambda f: setattr(f.config.compilation_config, 'cudagraph_mode', 'piecewise'),
        lambda f: setattr(f.runner.cudagraph_manager, '_graphs_captured', False),
        lambda f: setattr(f.runner.cudagraph_manager, 'graphs', {}),
    ]
    with patch.object(mod, '_runtime', side_effect=AssertionError('unselected runtime import')):
        assert not mod.install(NS())
        for mutation in mutations:
            f = fixture(); mutation(f)
            with patch.dict(sys.modules, f.modules): assert not mod.install(f.worker)
    for head_on in (False, True):
        f = fixture()
        if head_on: f.attached._head_graphs = object()
        else: setattr(f.head, f.native._STATE_NAME, None)
        with patch.dict(sys.modules, f.modules), patch.object(mod, '_runtime', return_value=(f.torch, f.native)), \
             patch.object(mod, 'Binding', side_effect=AssertionError('unselected binding')):
            assert not mod.install(f.worker)


def test_native_guard_device_fallback_calls_original_once():
    f = fixture(); calls = []; result = object()
    with patch.dict(sys.modules, f.modules):
        guard = mod.make_guard(f.torch, lambda tensor: calls.append(tensor) or result,
                               f.kernel.reduce_scale, f.device)
    tensor = f.Tensor()
    f.flags['current'] = 1
    assert guard(tensor) is result and calls == [tensor] and not f.launches
    f.flags['current'] = 0; tensor.device = NS(index=1)
    assert guard(tensor) is result and calls == [tensor, tensor] and not f.launches
    tensor.device = f.device; f.flags['capture'] = True
    assert guard(tensor) is result and calls == [tensor] * 3 and not f.launches


def test_native_lazy_sampling_proxy_inherits_attached_binding():
    f = fixture()
    del f.runner._mach_sampling_proxy
    del f.runner._mach_sampling_head
    with patch.dict(sys.modules, f.modules), patch.object(mod, '_runtime', return_value=(f.torch, f.native)), \
         patch.object(two_b, '_head_scale_kernel', f.kernel, create=True):
        assert mod.install(f.worker)
        # Execute the actual proxy state lookup used when normal sampling lazily
        # creates a proxy after startup. Native has no separate MXFP8 sidecar.
        path = Path(mod.__file__).with_name('sampling.py')
        code = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
        class_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'SamplingHead')
        getter = next(c for c in class_code.co_consts if isinstance(c, types.CodeType) and c.co_name == '_get_hybrid_lm_head_state')
        proxy = NS(state=None)
        lookup = types.FunctionType(getter, {})
        state = lookup(proxy, f.head)
        assert state is f.attached
        assert state.coarse_logits(f.Tensor(), None) is f.logits
        assert len(f.launches) == 1
        assert f.native.HybridNvfp4LmHead.coarse_logits is f.unrelated.coarse_logits.__func__
