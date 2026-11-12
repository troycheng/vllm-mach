# SPDX-License-Identifier: Apache-2.0
"""Fixed activation inputs for qualified Block-FP8 exact2048 graph islands."""
import threading
from contextlib import nullcontext


def _qualified_metadata_parameters():
    # Keep the same one-time source qualification as the tested prototype.
    import hashlib
    import inspect
    try:
        from vllm.model_executor import parameter
        source = inspect.getsourcefile(parameter)
        with open(source, 'rb') as handle:
            qualified = hashlib.sha256(handle.read()).hexdigest() == (
                '1a4bbd7400fd1ba79e8e1e8d19666cd22881ec83c35e6e01b3fdc78faee23171')
    except (ImportError, OSError, TypeError):
        qualified = False
    if not qualified:
        return (), None
    return ((parameter.ModelWeightParameter, parameter.BlockQuantScaleParameter),
            parameter.BasevLLMParameter.__torch_function__.__func__)


_METADATA_PARAMETER_TYPES = ()
_BASE_TORCH_FUNCTION = None
_METADATA_INITIALIZED = False


def _initialize_metadata_parameters():
    global _METADATA_PARAMETER_TYPES, _BASE_TORCH_FUNCTION, _METADATA_INITIALIZED
    if not _METADATA_INITIALIZED:
        _METADATA_PARAMETER_TYPES, _BASE_TORCH_FUNCTION = _qualified_metadata_parameters()
        _METADATA_INITIALIZED = True


def _metadata_context(t):
    import torch
    if (type(t) in _METADATA_PARAMETER_TYPES and
            getattr(type(t).__torch_function__, '__func__', None) is _BASE_TORCH_FUNCTION):
        return torch._C.DisableTorchFunctionSubclass()
    return nullcontext()

CORE=2048*4096
GATE=(2048-1)*12288+4096
RESIDUAL=2048*2560
BANK_BYTES=(CORE+GATE+RESIDUAL)*2
COPY_BYTES_PER_ISLAND=(CORE+CORE+RESIDUAL)*2


def role(t):
    import torch
    if t.dtype!=torch.bfloat16:return None
    shape,stride=tuple(t.shape),tuple(t.stride())
    if (shape,stride)in(((2048,32,128),(4096,128,1)),((2048,16,256),(4096,256,1))):return 'core'
    if (shape,stride)in(((2048,32,128),(12288,128,1)),((2048,4096),(4096,1))):return 'gate'
    if (shape,stride)==((2048,2560),(2560,1)):return 'residual'
    return None


class ActivationBank:
    def __init__(self,device):
        _initialize_metadata_parameters()
        import torch
        self.device=torch.device(device);self.capture_stream=torch.cuda.current_stream(self.device)
        self.stream=self.capture_stream.cuda_stream;self.thread=threading.get_ident()
        self.phase='capture';self.runtime_stream=None;self.runtime_thread=None
        self.buffers={name:torch.empty(count,dtype=torch.bfloat16,device=self.device)
                      for name,count in [('core',CORE),('gate',GATE),('residual',RESIDUAL)]}
        self.views={};self.bytes=sum(t.numel()*t.element_size()for t in self.buffers.values())
        assert self.bytes==BANK_BYTES

    def bind_runtime(self,main_stream):
        import torch
        assert self.phase=='capture'
        assert torch.cuda.current_stream(self.device).cuda_stream==main_stream.cuda_stream,'Warm must return to runner main stream'
        # One untimed handoff after original capture; strong bank references live
        # for the entire engine. Never switch streams inside real execution.
        main_stream.wait_stream(self.capture_stream)
        self.runtime_stream=main_stream.cuda_stream;self.phase='runtime'

    def reject(self,reason,detail=None):
        return None,reason

    def prepare(self,args,readonly,signature=None):
        import torch
        observed_stream=torch.cuda.current_stream(self.device).cuda_stream;observed_thread=threading.get_ident()
        expected_stream=self.stream if self.phase=='capture'else self.runtime_stream
        expected_thread=self.thread if self.phase=='capture'else self.runtime_thread
        if observed_stream!=expected_stream:
            return self.reject('stream_mismatch',dict(phase=self.phase,expected=expected_stream,observed=observed_stream,thread=observed_thread))
        if expected_thread is not None and observed_thread!=expected_thread:
            return self.reject('thread_mismatch',dict(phase=self.phase,expected=expected_thread,observed=observed_thread,stream=observed_stream))
        converted=list(args);seen=set();spec=[];copies=[]
        for i,t in enumerate(args):
            if not isinstance(t,torch.Tensor):
                if type(t)not in(int,float,bool,type(None)):return self.reject('unknown_scalar_type',dict(index=i,type=type(t).__name__))
                spec.append(('scalar',type(t).__name__,t));continue
            with _metadata_context(t):
                storage=(str(t.device),t.untyped_storage().data_ptr())
                if storage in readonly:
                    spec.append(('readonly',str(t.device),t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype)));continue
            name=role(t)
            if name is None or name in seen or t.device!=self.device:return self.reject('activation_layout_role_device',dict(index=i,role=name,shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device)))
            seen.add(name);key=(name,tuple(t.shape),tuple(t.stride()))
            if key not in self.views:self.views[key]=self.buffers[name].as_strided(t.shape,t.stride())
            converted[i]=self.views[key];copies.append((converted[i],t))
            spec.append(('activation',name,tuple(t.shape),tuple(t.stride()),str(t.dtype)))
        if seen!={'core','gate','residual'}:return self.reject('activation_roles',dict(observed=sorted(seen)))
        if signature is not None and signature!=spec:return self.reject('input_signature',dict(mismatches=[dict(index=i,expected=old,observed=new)for i,(old,new)in enumerate(zip(signature,spec))if old!=new],expected_args=len(signature),observed_args=len(spec)))
        # Only enqueue after complete admission; preserve exact source dtype and
        # view strides, all readonly objects, and original same-stream ordering.
        if self.phase=='runtime'and self.runtime_thread is None:self.runtime_thread=observed_thread
        for dst,src in copies:dst.copy_(src,non_blocking=True)
        return (tuple(converted),spec),None
