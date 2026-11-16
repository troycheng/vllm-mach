# SPDX-License-Identifier: Apache-2.0
"""Selected post-capture 2B global-scale binding with the frozen guard."""
import ast
import hashlib
from pathlib import Path
import types

_STATE = "_mach_2b_head_scale"
_ORIGINAL_AST_SHA = "50ec5c08edf91f5528b1431b7fb3194f588804b75032e21c8cbb237369bfaf10"


def _bind(torch,reference):
    from ._head_scale_kernel import reduce_scale
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from torch.overrides import _get_current_function_mode
    is_inference=torch.is_inference_mode_enabled;is_grad=torch.is_grad_enabled
    is_compiling=torch.compiler.is_compiling;is_tracing=torch.jit.is_tracing;is_scripting=torch.jit.is_scripting
    is_capture=torch.cuda.is_current_stream_capturing;zero_tensor=getattr(torch,'_is_zerotensor',None)
    def eligible(tensor):
        if (_get_current_dispatch_mode()is not None or _get_current_function_mode()is not None
                or not is_inference()or is_grad()or is_compiling()or is_tracing()or is_scripting()):return False
        if type(tensor)is not torch.Tensor:return False
        if not tensor.is_cuda or tensor.dtype!=torch.bfloat16:return False
        if tuple(tensor.shape)!=(16,2048)or tuple(tensor.stride())!=(2048,1):return False
        if tensor.is_conj()or tensor.is_neg()or(zero_tensor is not None and zero_tensor(tensor)):return False
        if is_capture():return False
        return True
    def guarded(tensor):
        if not eligible(tensor):return reference(tensor)
        output=torch.empty((),dtype=torch.float32,device=tensor.device)
        reduce_scale[(1,)](tensor,output,num_warps=8,enable_fp_fusion=False)
        return output
    guarded.eligible=eligible
    return guarded


def _selected(worker):
    contract = getattr(worker, "_mach_2b_contract", {})
    if (contract.get("mode") != "production" or contract.get("ssm_dtype") != "float32"
            or contract.get("hybrid_greedy_head") is not True):
        return False
    installed = getattr(worker, "_mach_2b_hybrid_head", None)
    state = installed.get("state") if type(installed) is dict else None
    if state is None or state.candidates != 128 or state._head_graphs is not None:
        return False
    runner = worker.model_runner
    parallel = worker.vllm_config.parallel_config
    if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or runner.speculative_config is not None):
        return False
    manager = runner.cudagraph_manager
    return bool(manager is not None and manager._graphs_captured and manager.graphs)


def _runtime():
    import torch
    from vllm_mach.mxfp6 import hybrid_nvfp4_lm_head
    return torch, hybrid_nvfp4_lm_head


def _original(module):
    original = module._global_scale
    if (type(original) is not types.FunctionType or original.__closure__
            or hasattr(original, "__wrapped__") or original.__name__ != "_global_scale"
            or original.__module__ != module.__name__
            or original.__globals__ is not vars(module)
            or module.HybridNvfp4LmHead.coarse_logits.__globals__ is not vars(module)
            or module._NVFP4_MAX_VALUE != 2688.0):
        raise RuntimeError("2B head scale original global owner changed")
    path = Path(module.__file__).resolve()
    if path != Path(original.__code__.co_filename).resolve():
        raise RuntimeError("2B head scale original source owner changed")
    source = path.read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_global_scale")
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    compiled = compile(source, original.__code__.co_filename, "exec", dont_inherit=True)
    code = next(c for c in compiled.co_consts
                if isinstance(c, types.CodeType) and c.co_name == "_global_scale")
    if digest != _ORIGINAL_AST_SHA or code != original.__code__:
        raise RuntimeError("2B head scale original source changed")
    return original


def install(worker):
    if not _selected(worker):
        return False
    if hasattr(worker, _STATE):
        raise RuntimeError("2B head scale is already installed")
    torch, module = _runtime()
    original = _original(module)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("2B head scale requires completed capture")
    guarded = _bind(torch, original)
    if module._global_scale is not original:
        raise RuntimeError("2B head scale global owner changed before installation")
    module._global_scale = guarded
    setattr(worker, _STATE, {"installed": True, "original_AST_sha256": _ORIGINAL_AST_SHA,
                            "rows": 16, "columns": 2048, "num_warps": 8,
                            "enable_fp_fusion": False})
    return True


def inspect_worker(worker):
    return dict(getattr(worker, _STATE, {"installed": False}))
