# SPDX-License-Identifier: Apache-2.0
"""Post-capture, state-local scale binding for the native MoE FP16 profile.

The M16 kernel and complete guard are the qualified native caller. Only the
selected head states receive a private coarse_logits function; the module
scale, model graphs, collectives, and other head instances stay untouched.
"""
import hashlib
from pathlib import Path
import types

HEAD_SHA = '6a71f278e93b1e7d5414d3595fac465cddf06437a55db5d2678788364249b025'
KERNEL_SHA = 'e1617aff5de9313994a018cc65095b7367e73c20ea66f43c8c692cfe6968c38c'


def checked_function(module, name, expected_sha):
    path = Path(module.__file__).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise RuntimeError('Native scale source identity changed: ' + str(path))
    function = getattr(module, name)
    function = function.fn if hasattr(function, 'fn') else function
    if (type(function) is not types.FunctionType or function.__closure__
            or function.__globals__ is not vars(module)
            or Path(function.__code__.co_filename).resolve() != path):
        raise RuntimeError('Native scale function owner changed: ' + name)
    code = next(c for c in compile(source, str(path), 'exec', dont_inherit=True).co_consts
                if isinstance(c, types.CodeType) and c.co_name == name)
    if code != function.__code__:
        raise RuntimeError('Native scale code differs from frozen source: ' + name)
    return function


def make_guard(torch, reference, kernel, owned_device):
    """Full hot guard, including TP-process device ownership; no observation."""
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from torch.overrides import _get_current_function_mode
    is_inference = torch.is_inference_mode_enabled
    is_grad = torch.is_grad_enabled
    is_compiling = torch.compiler.is_compiling
    is_tracing = torch.jit.is_tracing
    is_scripting = torch.jit.is_scripting
    current_device = torch.cuda.current_device
    is_capture = torch.cuda.is_current_stream_capturing
    zero_tensor = getattr(torch, '_is_zerotensor', None)

    def eligible(tensor):
        if (_get_current_dispatch_mode() is not None or _get_current_function_mode() is not None
                or not is_inference() or is_grad() or is_compiling() or is_tracing() or is_scripting()):
            return False
        if type(tensor) is not torch.Tensor:
            return False
        if not tensor.is_cuda or tensor.dtype != torch.bfloat16:
            return False
        if tuple(tensor.shape) != (16, 2048) or tuple(tensor.stride()) != (2048, 1):
            return False
        if tensor.device != owned_device or current_device() != owned_device.index:
            return False
        if tensor.is_conj() or tensor.is_neg() or (zero_tensor is not None and zero_tensor(tensor)):
            return False
        if is_capture():
            return False
        return True

    def guarded(tensor):
        if not eligible(tensor):
            return reference(tensor)
        output = torch.empty((), dtype=torch.float32, device=tensor.device)
        kernel[(1,)](tensor, output, num_warps=8, enable_fp_fusion=False)
        return output

    guarded.eligible = eligible
    return guarded


class Binding:
    """Private state-method binding; global scale and method bytecode stay intact."""
    def __init__(self, worker, torch, native, kernel_module):
        self.reference = checked_function(native, '_global_scale', HEAD_SHA)
        checked_function(kernel_module, 'reduce_scale', KERNEL_SHA)
        checked_function(kernel_module, 'scalar_finalize', KERNEL_SHA)
        if native._NVFP4_MAX_VALUE != 2688.0:
            raise RuntimeError('Native scale constant changed')
        runner = worker.model_runner
        manager = runner.cudagraph_manager
        if manager is None or not manager._graphs_captured or not manager.graphs:
            raise RuntimeError('Native scale binding requires completed ordinary capture')
        # Actual Qwen3.5 MoE multimodal owner chain, also used by native
        # sampling.try_sample; no guessed direct-head or wrapper fallback.
        from vllm.model_executor.models import qwen3_5
        outer = runner.model
        if type(outer) is not qwen3_5.Qwen3_5MoeForConditionalGeneration:
            raise RuntimeError('Recorded MoE outer model class changed')
        language = outer.language_model
        if type(language) is not qwen3_5.Qwen3_5MoeForCausalLM:
            raise RuntimeError('Recorded MoE language-model owner changed')
        backbone = language.model
        if not isinstance(backbone, qwen3_5.Qwen3_5Model):
            raise RuntimeError('Recorded MoE backbone owner changed')
        head = language.lm_head
        processor = language.logits_processor
        if worker.vllm_config.parallel_config.tensor_parallel_size != 2 or head.tp_size != 2:
            raise RuntimeError('Native head scale requires the qualified TP2 owner')
        attached = native.get_hybrid_nvfp4_lm_head(head)
        states = [attached]
        proxy = getattr(runner, '_mach_sampling_proxy', None)
        if proxy is not None:
            saved_head, saved_ptr = runner._mach_sampling_head
            if proxy.processor is not processor or saved_head is not head or saved_ptr != head.weight.data_ptr():
                raise RuntimeError('Existing native sampling proxy belongs to another head')
            states.append(proxy._get_hybrid_lm_head_state(head))
        # A distinct existing sampling state is included, not rejected by sampling mode.
        self.states = list({id(state): state for state in states if state is not None}.values())
        if attached is None or not self.states:
            raise RuntimeError('No actual native hybrid head state')
        owned_device = head.weight.device
        if owned_device.index is None or torch.cuda.current_device() != owned_device.index:
            raise RuntimeError('Native TP worker/current device mismatch')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Native caller binding cannot occur during capture')
        original_method = native.HybridNvfp4LmHead.coarse_logits
        path = Path(native.__file__).resolve()
        module_code = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
        class_code = next(c for c in module_code.co_consts
                          if isinstance(c, types.CodeType) and c.co_name == 'HybridNvfp4LmHead')
        method_code = next(c for c in class_code.co_consts
                           if isinstance(c, types.CodeType) and c.co_name == 'coarse_logits')
        if (type(original_method) is not types.FunctionType or original_method.__closure__
                or original_method.__globals__ is not vars(native)
                or original_method.__code__ != method_code):
            raise RuntimeError('Native coarse method differs from the frozen source')
        for state in self.states:
            if (type(state) is not native.HybridNvfp4LmHead or state.input_size != 2048
                    or state._head_graphs is not None or state.weight.device != owned_device
                    or 'coarse_logits' in vars(state)
                    or state.coarse_logits.__func__ is not original_method):
                raise RuntimeError('Native head state/caller/private-graph identity changed')
        self.native = native
        self.kernel_module = kernel_module
        self.original_method = original_method
        self.guarded = make_guard(torch, self.reference, kernel_module.reduce_scale, owned_device)
        self.installed = []
        self.clone = None
        self.runner, self.head, self.torch = runner, head, torch
        self.outer, self.language, self.backbone, self.processor = outer, language, backbone, processor
        self.expected_identity = self.identity()
        self.owner = dict(native_state_ids=[id(s) for s in self.states], head_id=id(head),
                          device=str(owned_device), tp_size=2,
                          no_sampling_mode_or_MTP_restriction=True)

    def identity(self):
        """Cold control-plane identity only; no tensor read or new model call."""
        manager = self.runner.cudagraph_manager
        def tensor_info(value):
            return (id(value), value.data_ptr(), tuple(value.shape),
                    tuple(value.stride()), str(value.dtype), str(value.device))
        current = [self.native.get_hybrid_nvfp4_lm_head(self.head)]
        proxy = getattr(self.runner, '_mach_sampling_proxy', None)
        if proxy is not None:
            current.append(proxy._get_hybrid_lm_head_state(self.head))
        outer = self.runner.model
        language = outer.language_model
        backbone = language.model
        return dict(
            owner_chain=dict(outer=id(outer), outer_class=type(outer).__qualname__,
                language=id(language), language_class=type(language).__qualname__,
                backbone=id(backbone), backbone_class=type(backbone).__qualname__,
                head=id(language.lm_head), processor=id(language.logits_processor),
                layer_container=id(backbone.layers), decoder_layers=[id(x) for x in backbone.layers],
                outer_get_top_tokens=(id(outer.get_top_tokens.__func__), id(outer.get_top_tokens.__func__.__code__)),
                language_get_top_tokens=(id(language.get_top_tokens.__func__), id(language.get_top_tokens.__func__.__code__)),
                processor_get_top_tokens=(id(language.logits_processor.get_top_tokens.__func__),
                                          id(language.logits_processor.get_top_tokens.__func__.__code__)),
                sampling_proxy=id(proxy), sampling_proxy_processor=id(proxy.processor) if proxy is not None else None),
            state_ids=sorted({id(s) for s in current if s is not None}),
            class_method=id(self.native.HybridNvfp4LmHead.coarse_logits),
            class_method_code=id(self.original_method.__code__),
            global_scale=id(self.native._global_scale),
            global_scale_code=id(self.reference.__code__),
            kernel={name: (id(getattr(self.kernel_module, name)),
                           id(getattr(self.kernel_module, name).fn.__code__))
                    for name in ('reduce_scale', 'scalar_finalize')},
            caller_globals={name: id(self.original_method.__globals__[name])
                            for name in self.original_method.__code__.co_names
                            if name in self.original_method.__globals__},
            head=id(self.head), weight=tensor_info(self.head.weight),
            graph_manager=id(manager), captured=manager._graphs_captured,
            graphs=sorted((repr(key), id(value)) for key, value in manager.graphs.items()),
            states=[dict(id=id(s), private_graph=id(s._head_graphs),
                         weight=tensor_info(s.weight), scale=tensor_info(s.scale),
                         global_scale=tensor_info(s.global_scale),
                         candidates=s.candidates, backend=s.backend,
                         input_size=s.input_size, output_size=s.output_size,
                         max_rows=s.max_rows) for s in self.states])

    def check(self):
        if self.identity() != self.expected_identity:
            raise RuntimeError('Native head/graph/source ownership changed')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Native caller control cannot occur during capture')
        for state in self.states:
            expected = self.clone if any(state is s for s in self.installed) else self.original_method
            if state.coarse_logits.__func__ is not expected:
                raise RuntimeError('Native coarse method owner changed')

    def bind_scale(self, function):
        """Call only after the original workload is drained; no model call here."""
        self.check()
        if self.installed:
            raise RuntimeError('Detach the prior binding first')
        if self.native._global_scale is not self.reference:
            raise RuntimeError('Module-global scale owner changed')
        for state in self.states:
            if state._head_graphs is not None or 'coarse_logits' in vars(state):
                raise RuntimeError('Native state was changed after binding preparation')
        self.clone = types.FunctionType(self.original_method.__code__,
            {**self.original_method.__globals__, '_global_scale': function},
            self.original_method.__name__, self.original_method.__defaults__,
            self.original_method.__closure__)
        if any(value is not self.clone.__globals__[name]
               for name, value in self.original_method.__globals__.items()
               if name != '_global_scale'):
            raise RuntimeError('Native caller clone changed another global')
        self.clone.__kwdefaults__ = self.original_method.__kwdefaults__
        self.clone.__annotations__ = self.original_method.__annotations__
        for state in self.states:
            state.coarse_logits = types.MethodType(self.clone, state)
            self.installed.append(state)

    def on(self):
        self.bind_scale(self.guarded)

    def detach(self):
        self.check()
        for state in self.installed:
            if state.coarse_logits.__func__ is not self.clone:
                raise RuntimeError('Refuse to overwrite a foreign native caller mutation')
        for state in self.installed:
            del state.coarse_logits
        self.installed.clear()
        self.clone = None
        if self.native._global_scale is not self.reference:
            raise RuntimeError('Module-global scale must remain unchanged')
        if any(s.coarse_logits.__func__ is not self.original_method for s in self.states):
            raise RuntimeError('Native caller failed to detach')


_STATE = "_mach_native_head_scale"


def _selected(worker):
    config = getattr(worker, "vllm_config", None)
    if config is None:
        return False
    model = config.model_config
    text = model.hf_text_config
    if (not worker.use_v2_model_runner or model.enforce_eager
            or config.cache_config.mamba_ssm_cache_dtype != "float16"
            or text.model_type != "qwen3_5_moe_text" or text.hidden_size != 2048
            or config.parallel_config.tensor_parallel_size != 2):
        return False
    from vllm.config import CompilationMode, CUDAGraphMode
    compilation = config.compilation_config
    if (compilation.mode != CompilationMode.VLLM_COMPILE
            or compilation.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE):
        return False
    manager = worker.model_runner.cudagraph_manager
    return bool(manager is not None and manager._graphs_captured and manager.graphs)


def _runtime():
    import torch
    from . import hybrid_nvfp4_lm_head
    return torch, hybrid_nvfp4_lm_head


def install(worker):
    """Called once after ordinary model capture, without a model or kernel call."""
    if not _selected(worker):
        return False
    if hasattr(worker, _STATE):
        raise RuntimeError("Native head scale is already installed")
    torch, native = _runtime()
    # The native Qwen3.5 wrapper owns its head through language_model.
    head = worker.model_runner.model.language_model.lm_head
    state = native.get_hybrid_nvfp4_lm_head(head)
    if state is None or state._head_graphs is not None:
        return False
    from vllm_mach.mxfp8.two_b import _head_scale_kernel
    binding = Binding(worker, torch, native, _head_scale_kernel)
    binding.on()
    setattr(worker, _STATE, binding)
    return True


def inspect_worker(worker):
    """Installation receipt; no observation is attached to the hot path."""
    binding = getattr(worker, _STATE, None)
    if binding is None:
        return {"installed": False}
    return dict(installed=bool(binding.installed), rows=16, columns=2048,
                num_warps=8, enable_fp_fusion=False,
                head_source_sha256=HEAD_SHA, kernel_source_sha256=KERNEL_SHA,
                **binding.owner)
