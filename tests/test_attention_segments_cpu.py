"""Profile admission for the original fixed attention partitioning; no GPU."""
import ast,copy,os,unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
from vllm_mach.runtime import attention

def config():
 weight=dict(dtype='fp6_e3m2',group_size=32,qscheme='per_group',scale_format='e8m0',is_dynamic=False,symmetric=True)
 activation=dict(dtype='fp8_e4m3',group_size=32,qscheme='per_group',scale_format='e8m0',is_dynamic=True,symmetric=True)
 return N(model_config=N(dtype='torch.bfloat16',quantization='quark',hf_text_config=N(model_type='qwen3_5_moe_text'),model_arch_config=N(quantization_config={'global_quant_config':{'weight':weight,'input_tensors':activation}})),cache_config=N(cache_dtype='auto'))

class AttentionPartitionProfiles(unittest.TestCase):
 def setUp(self):
  self.env=patch.dict(os.environ,{},clear=True);self.env.start();self.addCleanup(self.env.stop)
  self.registered=patch.object(attention,'_NATIVE_REGISTERED',False);self.registered.start();self.addCleanup(self.registered.stop)
 def test_native_requires_registered_selector_and_exact_supported_format(self):
  cfg=config();self.assertFalse(attention.preserve_old_softmax_segments(cfg));attention.enable_native()
  for model in('qwen3_5_text','qwen3_5_moe_text'):
   cfg.model_config.hf_text_config.model_type=model;self.assertTrue(attention.preserve_old_softmax_segments(cfg))
  for field,value in(('dtype','fp8_e4m3'),('group_size',64),('qscheme','per_tensor'),('scale_format','float'),('is_dynamic',True),('symmetric',False)):
   bad=copy.deepcopy(cfg);bad.model_config.model_arch_config.quantization_config['global_quant_config']['weight'][field]=value;self.assertFalse(attention.preserve_old_softmax_segments(bad))
  cfg.model_config.quantization='compressed-tensors';self.assertFalse(attention.preserve_old_softmax_segments(cfg))
 def test_native_marker_follows_actual_plugin_registration_result(self):
  source=Path(__file__).resolve().parents[1]/'src/vllm_mach/plugin.py'
  fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='register')
  for registered in(False,True):
   attention._NATIVE_REGISTERED=False
   ns=dict(__package__='vllm_mach',os=os,version=lambda _: '0.33.0',BASE_PACKAGES={'vllm':'0.33.0'},register_dense_kernel=lambda:registered,register_compile_factors=lambda:None)
   exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns);ns['register']()
   self.assertEqual(attention._NATIVE_REGISTERED,registered)
 def test_only_explicit_two_b_profile_and_BF16_cache(self):
  cfg=config()
  with patch.dict(os.environ,{'VLLM_MACH_PROFILE':'qwen35-2b-mxfp8-champion-v1'}):
   self.assertTrue(attention.preserve_old_softmax_segments(cfg))
   for dtype in('fp8','fp8_e4m3','fp8_e5m2','int8_per_token_head'):
    cfg.cache_config.cache_dtype=dtype;self.assertFalse(attention.preserve_old_softmax_segments(cfg))
  cfg.cache_config.cache_dtype='auto';attention.enable_native()
  for profile in('qwen35-4b-mxfp8-champion-v1','qwen35-4b-block-fp8-v1','unknown'):
   with patch.dict(os.environ,{'VLLM_MACH_PROFILE':profile}):self.assertFalse(attention.preserve_old_softmax_segments(cfg))
 def test_stock_and_unsupported_native_keep_upstream_policy(self):
  cfg=config();attention.enable_native()
  for field,value in(('model_type','other'),):
   setattr(cfg.model_config.hf_text_config,field,value);self.assertFalse(attention.preserve_old_softmax_segments(cfg))
  cfg=config();cfg.model_config.dtype='torch.float16';self.assertFalse(attention.preserve_old_softmax_segments(cfg))
  cfg=config();cfg.model_config.model_arch_config.quantization_config=None;self.assertFalse(attention.preserve_old_softmax_segments(cfg))

if __name__=='__main__':unittest.main()
