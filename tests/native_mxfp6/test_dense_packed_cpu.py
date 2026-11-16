"""Dense overlap admission and actual kernel arguments; no Torch/GPU dependency."""
import contextlib,os,sys,unittest
from types import SimpleNamespace as N
from unittest.mock import patch
from vllm_mach.mxfp6.gdn import dense_packed,legacy

def contiguous(shape):
 stride=[];value=1
 for n in reversed(shape):stride.append(value);value*=n
 return tuple(reversed(stride))
class Tensor:
 def __init__(self,shape,dtype,device='cuda:0',stride=None):
  self.shape,self.dtype,self.device=shape,dtype,device;self.ndim=len(shape);self.is_cuda=device.startswith('cuda');self._stride=stride or contiguous(shape)
 def stride(self):return self._stride
 def is_contiguous(self):return self._stride==contiguous(self.shape)
class DensePackedTests(unittest.TestCase):
 def setUp(self):
  self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close);self.bf16,self.fp32,self.fp16,self.i32=object(),object(),object(),object();self.calls=[];self.capturing=False
  torch=N(bfloat16=self.bf16,float32=self.fp32,float16=self.fp16,int32=self.i32,cuda=N(is_current_stream_capturing=lambda:self.capturing))
  torch.zeros=lambda shape,*,dtype,device:Tensor(shape,dtype,device);torch.empty=torch.zeros
  torch.empty_strided=lambda shape,stride,*,dtype,device:Tensor(shape,dtype,device,stride)
  torch.full=lambda shape,value,*,dtype,device:Tensor(shape,dtype,device)
  for p in[patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,{'torch':torch}),patch.object(dense_packed,'_WARMED',set()),patch.object(dense_packed,'_STATS',dict(prepared=[],calls={},capture_calls={},kernels={},warmups={})),patch.object(legacy,'_ENABLED',False)]:self.stack.enter_context(p)
  calls=self.calls
  class Kernel:
   def __getitem__(self,grid):
    def launch(*args,**kwargs):
     calls.append((grid,args,kwargs));return N(hash='one-qualified-kernel',metadata=N(num_warps=1,num_stages=3,ptx_version=90),asm={'ptx':'PTX'})
    return launch
  self.stack.enter_context(patch.object(dense_packed,'_kernel',return_value=Kernel()))
 def layer(self,dtype=None):return N(prefix='model.layers.0.linear_attn',_mach_dense_norm=True,_mach_gdn_prepared=True,get_state_dtype=lambda:(self.bf16,dtype or self.fp32),gdn_prefill_backend='triton',tp_size=2,num_k_heads=16,num_v_heads=48,A_log=Tensor((24,),self.fp32),dt_bias=Tensor((24,),self.bf16))
 def args(self,m=32):return [Tensor((m,5120),self.bf16,stride=(8192,1)),Tensor((m,24),self.bf16),Tensor((m,24),self.bf16),Tensor((130,24,128,128),self.fp32,stride=(401408,16384,128,1)),Tensor((m,1,24,128),self.bf16),Tensor((m,),self.i32)]
 def test_real_legacy_prepare_warms_before_capture_without_kv_cache(self):
  layer=self.layer();self.assertFalse(hasattr(layer,'kv_cache'));legacy.prepare(layer);self.assertTrue(layer._mach_dense_fp32_packed)
  self.assertEqual(len(self.calls),1);self.assertEqual(dense_packed._STATS['calls'],{});self.assertEqual(dense_packed._STATS['capture_calls'],{});self.assertEqual(dense_packed._STATS['warmups'],{'cuda:0':1})
  layer2=self.layer();legacy.prepare(layer2);legacy.prepare(layer2);self.assertEqual(len(self.calls),1);self.assertFalse(legacy.enabled())
 def test_fp16_moe_unprepared_and_explicit_other_backend_remain_unmodified(self):
  cases=[self.layer(self.fp16),self.layer(),self.layer(),self.layer(),self.layer()]
  cases[1]._mach_dense_norm=False;cases[2]._mach_gdn_prepared=False;cases[3].gdn_prefill_backend='flashinfer';cases[4].num_v_heads=32
  for layer in cases:
   with self.subTest(layer=layer):dense_packed.prepare(layer);self.assertFalse(getattr(layer,'_mach_dense_fp32_packed',False));self.assertFalse(dense_packed.decode(layer,*self.args()))
  self.assertFalse(self.calls)
 def test_exact_downstream_alias_kwargs_rows_capture_and_unchanged_fallback(self):
  layer=self.layer();legacy.prepare(layer);self.calls.clear()
  for m in(16,24,32):
   args=self.args(m);self.capturing=True;self.assertTrue(legacy.decode(layer,*args));grid,passed,kw=self.calls[-1]
   self.assertEqual(grid,(4,m*24));self.assertIs(passed[5],args[4]);self.assertIs(passed[6],args[3]);self.assertIs(passed[7],args[3]);self.assertIs(passed[8],args[5]);self.assertEqual(passed[9],128**-.5)
   self.assertEqual(kw,dict(stride_mixed_qkv_tok=8192,stride_a_tok=24,stride_b_tok=24,stride_init_state_token=401408,stride_final_state_token=401408,stride_indices_seq=1,H=8,HV=24,K=128,V=128,BK=128,BV=32,SOFTPLUS_THRESHOLD=20.,USE_QK_L2NORM_IN_KERNEL=True,SPLIT_BATCH_HEAD_GRID=False,num_warps=1,num_stages=3,ptx_version=90))
  before=len(self.calls)
  for m in(1,2,4,8,48,64):self.assertFalse(legacy.decode(layer,*self.args(m)))
  for index,replacement in[(0,Tensor((32,5120),self.bf16)),(3,Tensor((130,24,128,128),self.fp16,stride=(409600,16384,128,1))),(5,Tensor((32,),self.fp32))]:
   args=self.args();args[index]=replacement;self.assertFalse(legacy.decode(layer,*args))
  self.assertEqual(len(self.calls),before);self.assertEqual(sum(dense_packed._STATS['calls'].values()),3);self.assertEqual(sum(dense_packed._STATS['capture_calls'].values()),3)
if __name__=='__main__':unittest.main()
