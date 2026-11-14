# SPDX-License-Identifier: Apache-2.0
"""Profile-local substitution of the exact generated CUDA quantization call."""
import ast,collections,functools,hashlib,importlib.metadata as md,inspect,json,sys
from pathlib import Path
HERE=Path(__file__).parent
_INSTALLED=False
_ORIGINAL_OP=None
_MODULES={}
_COUNTS=collections.Counter()
_SHAPES=collections.Counter()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def eligible(x,q,s,g,eps,minimum,maximum,ue8,col,tma):
 import torch
 return (g==128 and eps==1e-10 and minimum==-448. and maximum==448.
         and ue8 is False and col is True and tma is False
         and x.ndim==2 and x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()
         and x.shape[1]%128==0 and q.shape==x.shape and q.dtype==torch.float8_e4m3fn
         and q.device==x.device and q.is_contiguous() and s.ndim==2
         and s.shape==(x.shape[0],x.shape[1]//128) and s.dtype==torch.float32
         and s.device==x.device and s.stride()==(1,x.shape[0]))

def dispatch(x,q,s,g,eps,minimum,maximum,ue8,col,tma):
 if eligible(x,q,s,g,eps,minimum,maximum,ue8,col,tma):
  from ._quantize_kernel import quantize_out
  _COUNTS['eligible']+=1;_SHAPES[str(tuple(x.shape))]+=1
  quantize_out(x,q,s)
  return None
 _COUNTS['fallback']+=1
 return _ORIGINAL_OP(x,q,s,g,eps,minimum,maximum,ue8,col,tma)

class _Delegate:
 def __init__(self,original,attribute,replacement):
  self._original=original;self._attribute=attribute;self._replacement=replacement
 def __getattr__(self,name):
  return self._replacement if name==self._attribute else getattr(self._original,name)

def bind(module,path):
 global _ORIGINAL_OP
 import torch
 if getattr(module,'_mach_fp8_division_bound',False):return module
 source=Path(path).read_text()
 if 'torch.ops._C.per_token_group_fp8_quant.default'not in source:return module
 calls=[n for n in ast.walk(ast.parse(source))if isinstance(n,ast.Call)and ast.unparse(n.func)=='torch.ops._C.per_token_group_fp8_quant.default']
 if not calls:return module
 # Only the observed ten-argument selector enters this profile-local module path.
 if any(len(n.args)!=10 or n.keywords for n in calls):raise RuntimeError('Quant callsite source contract changed')
 assert module.torch is torch
 packet=torch.ops._C.per_token_group_fp8_quant
 if _ORIGINAL_OP is None:_ORIGINAL_OP=packet.default
 assert packet.default is _ORIGINAL_OP,'Global op must remain untouched'
 qp=_Delegate(packet,'default',dispatch)
 cp=_Delegate(torch.ops._C,'per_token_group_fp8_quant',qp)
 op=_Delegate(torch.ops,'_C',cp)
 module.torch=_Delegate(torch,'ops',op);module._mach_fp8_division_bound=True
 _MODULES[str(path)]={'source_sha256':sha(path),'matched_calls':len(calls)}
 return module

def install():
 global _INSTALLED
 if _INSTALLED:return
 from .profile import NAME
 import os
 if os.environ.get('VLLM_MACH_PROFILE') != NAME:return False
 from torch._inductor.codecache import PyCodeCache
 spec=json.loads((HERE/'data'/'legacy_math_sources.json').read_text())
 assert md.version('vllm')=='0.33.0'and md.version('torch')=='2.14.0+cu130'and md.version('triton')=='3.8.0'
 path=Path(md.distribution('vllm').locate_file('vllm/model_executor/layers/quantization/utils/fp8_utils.py'))
 assert sha(path)==spec['fp8_utils_sha256'],'Quant frontend source drift'
 original=PyCodeCache.load_by_key_path.__func__
 assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==spec['PyCodeCache.load_by_key_path_sha256'],'Module loader source drift'
 @functools.wraps(original)
 def load(cls,*args,**kwargs):
  module=original(cls,*args,**kwargs)
  return bind(module,kwargs.get('path',args[1]if len(args)>1 else module.__file__))
 PyCodeCache.load_by_key_path=classmethod(load);_INSTALLED=True

def receipt(require=True):
 if require:assert _INSTALLED and _MODULES and _COUNTS['eligible']>0,(_INSTALLED,_MODULES,_COUNTS)
 from .guard_norm import cache_receipt
 kernel=sys.modules.get(__package__+'._quantize_kernel')
 return {'actual_compiled':[]if kernel is None else cache_receipt(kernel._quant),'installed':_INSTALLED,'modules':_MODULES,'calls':dict(_COUNTS),'eligible_shapes':dict(_SHAPES),
         'sources':{p.name:sha(p)for p in [Path(__file__),HERE/'_quantize_kernel.py']},
         'scope':'Actual Python launches at warmup/capture/eager prefill; CUDA graph replays do not increment Python counters',
         'global_op_unchanged':True,'changes':'Only eligible exact _C quant output math; original allocations/strides/unknown-layout fallback retained'}
