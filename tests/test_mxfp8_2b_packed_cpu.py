"""Scope and exact source transform of the 2B FP32 stock low-M decoder."""
import ast
import importlib.util
import inspect
import functools
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]/'src/vllm_mach/mxfp8/two_b/gdn'
N=types.SimpleNamespace

def load(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class PackedScope(unittest.TestCase):
 def test_low_rows_warm_bound_pool_with_only_zero_slots_before_capture(self):
  worker=load('packed_warm_worker','worker.py');worker.__package__='vllm_mach.mxfp8.two_b.gdn'
  calls=[]
  def empty(shape,**kw):return N(shape=shape,**kw)
  def zeros(shape,**kw):return N(shape=shape,all_zero=True,**kw)
  torch=N(float32='fp32',bfloat16='bf16',int32='i32',empty=empty,zeros=zeros,
          empty_strided=lambda shape,stride,**kw:N(shape=shape,stride=stride,**kw))
  state=N(dtype='fp32',device='cuda',stride=(278528,16384,128,1));layer=N(A_log=object(),dt_bias=object())
  module=types.ModuleType('vllm_mach.mxfp8.two_b.gdn.packed_fp32')
  def decode(q,a,b,A,dt,base,out,ids):
   self.assertIs(base,state);self.assertIs(A,layer.A_log);self.assertIs(dt,layer.dt_bias)
   self.assertTrue(ids.all_zero);self.assertEqual(ids.dtype,'i32')
   self.assertEqual(q.stride,(8192,1));self.assertEqual(q.shape,(ids.shape[0],6144))
   self.assertEqual(out.shape,(ids.shape[0],1,16,128));calls.append(ids.shape[0])
  module.decode=decode
  with patch.dict(sys.modules,{'torch':torch,module.__name__:module}):
   worker._warm_stock_packed(layer,state,160);self.assertEqual(calls,[1,2,4,8,16,24])
   calls.clear();worker._warm_stock_packed(layer,state,4);self.assertEqual(calls,[1,2,4])
   calls.clear();state.dtype='bf16';worker._warm_stock_packed(layer,state,160);self.assertEqual(calls,[])
  tree=ast.parse((ROOT/'worker.py').read_text());prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_prepare_pools')
  text=ast.unparse(prepare);self.assertLess(text.index('_warm_stock_packed('),text.index('torch.cuda.synchronize('))

 def test_materialize_parent_conv_packed_and_context_restore(self):
  worker=load('packed_lifecycle_worker','worker.py');worker._READY=True;worker._LAYERS={'L':object()}
  events=[]
  class Tensor:
   def __init__(self,shape,stride,dtype='bf16'):self.shape=shape;self.pitch=stride;self.dtype=dtype;self.device='cuda'
   def stride(self):return self.pitch
   def __getitem__(self,index):return self
   def unsqueeze(self,index):return self
  state=Tensor((642,16,128,128),(278528,16384,128,1),'fp32')
  class IDs:
   dtype='i32';device='cuda'
   def numel(self):return 24
   def __getitem__(self,index):return self
  ids=IDs()
  md=N(non_spec_state_indices_tensor=ids,spec_sequence_masks=None,num_spec_decodes=0,num_prefills=0,num_decodes=4,num_actual_tokens=4)
  layer=N(prefix='L',kv_cache=(object(),state),conv1d=N(weight=N(size=lambda i:6144 if i==0 else 4,view=lambda *a:object()),bias=None),activation='silu',A_log=object(),dt_bias=object())
  worker._metadata=lambda self:md;worker._layout=lambda *a:True;worker._target=lambda *a:False;worker._relevant=lambda _:ids;worker._view_key=lambda _:0;worker._VIEW_SCRATCH={0:object()}
  torch=N(float32='fp32',bfloat16='bf16',int32='i32',cuda=N(is_current_stream_capturing=lambda:False))
  def packed(*args):events.append('packed')
  packed_module=types.ModuleType('vllm_mach.mxfp8.two_b.gdn.packed_fp32');packed_module.decode=packed
  utils=types.ModuleType('vllm.model_executor.layers.mamba.mamba_utils');utils.is_conv_state_dim_first=lambda:True
  def convolution(x,*a,**kw):events.append('conv');return x
  def parent(self,q,b,a,out,hidden=None):events.append('parent');return scope['decode'](self,q,b,a,out,md)
  def original_decode(*a):events.append('original')
  tree=ast.parse((ROOT/'worker.py').read_text());register=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_register_gdn')
  functions=[n for n in register.body if isinstance(n,ast.FunctionDef)and n.name in('core','decode')]
  scope={**worker.__dict__, '__package__':'vllm_mach.mxfp8.two_b.gdn', 'torch':torch,
         '_ORIG_CORE':parent,'_ORIG_DECODE':original_decode,'functools':functools,
         'component':N(materialize_slots=lambda *a:events.append('materialize')),
         'gdn':N(causal_conv1d_update=convolution)}
  exec(compile(ast.Module(body=functions,type_ignores=[]),'actual-worker-functions','exec'),scope)
  with patch.dict(sys.modules,{'torch':torch,packed_module.__name__:packed_module,utils.__name__:utils}),patch.dict('os.environ',{'FLA_USE_FAST_OPS':'0'}):
   for m in(1,2,4,8,16,24):
    md.num_decodes=md.num_actual_tokens=m;events.clear();q=Tensor((m,6144),(8192,1));ba=Tensor((m,16),(16,1));out=Tensor((m,16,128),(2048,128,1))
    scope['core'](layer,q,ba,ba,out)
    self.assertEqual(events,['materialize','parent','conv','packed'])
    self.assertIsNone(worker._ACTIVE_PACKED.get())
   self.assertEqual(sum(worker._PACKED_EAGER.values()),6)
   counts=dict(worker._PACKED_EAGER)
   scope['_ORIG_CORE']=lambda *a: 'FI'
   self.assertEqual(scope['core'](layer,q,ba,ba,out),'FI')
   self.assertEqual(worker._PACKED_EAGER,counts)
   self.assertIsNone(worker._ACTIVE_PACKED.get())
   scope['_ORIG_CORE']=parent
   state.dtype='bf16';events.clear();scope['core'](layer,q,ba,ba,out)
   self.assertEqual(events,['materialize','parent','original'])
   state.dtype='fp32'
   def fail(*a):raise RuntimeError('device failure')
   packed_module.decode=fail
   with self.assertRaisesRegex(RuntimeError,'device failure'):scope['core'](layer,q,ba,ba,out)
   self.assertIsNone(worker._ACTIVE_PACKED.get())

 def test_only_prepared_FP32_supported_stock_rows(self):
  worker=load('packed_scope_worker','worker.py');worker._READY=True;worker._LAYERS={'L':object()}
  torch=N(float32='fp32',bfloat16='bf16',int32='i32')
  def t(shape,stride,dtype='bf16'):return N(shape=shape,stride=lambda:stride,dtype=dtype,device='cuda')
  def call(m,*,dtype='fp32',prefix='L',prefill=0,index_type='i32',pitch=8192):
   ids=N(dtype=index_type,device='cuda',numel=lambda:m)
   md=N(non_spec_state_indices_tensor=ids,spec_sequence_masks=None,num_spec_decodes=0,num_prefills=prefill,num_decodes=m,num_actual_tokens=m)
   return worker._packed_target(N(prefix=prefix),md,t((m,6144),(pitch,1)),t((m,16),(16,1)),t((m,16),(16,1)),N(dtype=dtype,device='cuda'))
  with patch.dict(sys.modules,{'torch':torch}),patch.dict('os.environ',{'FLA_USE_FAST_OPS':'0'}):
   for m in(1,2,4,8,16,24):self.assertTrue(call(m),m)
   for m in(3,32,48,64):self.assertFalse(call(m),m)
   for options in(dict(dtype='bf16'),dict(dtype='fp16'),dict(prefix='foreign'),dict(prefill=1),dict(index_type='i64'),dict(pitch=6144)):
    self.assertFalse(call(16,**options),options)
   with patch.dict('os.environ',{'FLA_USE_FAST_OPS':'1'}):self.assertFalse(call(16))

 def test_stock_kernel_transform_changes_only_four_reductions(self):
  module=load('vllm_mach.mxfp8.two_b.gdn.packed_fp32','packed_fp32.py')
  source='''def kernel(b_q,b_k,b_h,BV):
    q=tl.sum(b_q*b_q)
    k=tl.sum(b_k*b_k)
    v=tl.sum(b_h*b_k[None,:],1)
    o=tl.sum(b_h*b_q[None,:],1)
    return q,k,v,o
'''
  original=N(src=source,fn=N(__globals__={}),__name__='kernel')
  legacy=types.ModuleType('vllm_mach.mxfp8.gdn.legacy_reduction');legacy._old_square_128=object();legacy._old_dot_128=object()
  vu=types.ModuleType('vllm.triton_utils');vu.triton=N(jit=lambda fn:N(src=inspect.getsource(fn)))
  ops=types.ModuleType('vllm.third_party.flash_linear_attention.ops');ops.fused_recurrent=N(fused_recurrent_gated_delta_rule_packed_decode_kernel=original)
  with patch.dict(sys.modules,{'vllm.triton_utils':vu,'vllm.third_party.flash_linear_attention.ops':ops,legacy.__name__:legacy}):
   result=module._kernel();tree=ast.parse(result.src)
   names=[n.func.id for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
   self.assertEqual(names.count('_old_square_128'),2);self.assertEqual(names.count('_old_dot_128'),2)
   self.assertIs(module._kernel(),result)
   module._KERNEL=None;original.src=source.replace('tl.sum(b_q*b_q)','tl.sum(b_q)')
   with self.assertRaisesRegex(RuntimeError,'source changed'):module._kernel()

if __name__=='__main__':unittest.main()
