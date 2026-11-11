"""Native FP16 restoration admission; no CUDA/Torch import is required."""
import copy
import os
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch
from vllm_mach.mxfp6.gdn import legacy
from vllm_mach.runtime import attention


def config():
    weight = dict(dtype='fp6_e3m2', group_size=32, qscheme='per_group',
                  scale_format='e8m0', is_dynamic=False, symmetric=True)
    activation = dict(dtype='fp8_e4m3', group_size=32, qscheme='per_group',
                      scale_format='e8m0', is_dynamic=True, symmetric=True)
    text = N(model_type='qwen3_5_moe_text', hidden_size=2048, num_hidden_layers=40,
             linear_num_key_heads=16, linear_num_value_heads=32,
             linear_key_head_dim=128, linear_value_head_dim=128,
             num_attention_heads=16, num_key_value_heads=2, head_dim=256)
    return N(model_config=N(dtype='torch.bfloat16', quantization='quark', hf_text_config=text,
                           model_arch_config=N(quantization_config={'global_quant_config':
                           {'weight': weight, 'input_tensors': activation}})),
             parallel_config=N(tensor_parallel_size=2, pipeline_parallel_size=1,
                               decode_context_parallel_size=1, enable_dbo=False),
             cache_config=N(cache_dtype='auto', mamba_ssm_cache_dtype='float16'),
             speculative_config=None, additional_config={})


class NativeLegacyAdmission(unittest.TestCase):
    def setUp(self):
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'VLLM_QWEN3_5_FP16_SSM': '1'}, clear=True))
        self.stack.enter_context(patch.object(attention, '_NATIVE_REGISTERED', True))
        self.stack.enter_context(patch.object(legacy, '_ENABLED', False))
        self.stack.enter_context(patch.object(legacy, '_sm120', return_value=True))

    def test_default_and_explicit_backend_semantics(self):
        cfg = config()
        with patch.object(legacy, '_sm120', return_value=False), patch.object(legacy, 'install') as inactive:
            self.assertEqual(legacy.resolve_backend(cfg, 'auto'), 'auto')
            inactive.assert_not_called()
        with patch.object(legacy, 'install') as install:
            self.assertEqual(legacy.resolve_backend(cfg, 'auto'), 'triton')
            install.assert_called_once()
        for value in ('auto', 'flashinfer', 'cutedsl', 'unknown'):
            cfg.additional_config = {'gdn_prefill_backend': value}
            with patch.object(legacy, 'install') as install:
                self.assertEqual(legacy.resolve_backend(cfg, value), value)
                self.assertFalse(legacy.enabled())
                install.assert_not_called()
        cfg.additional_config = {'gdn_prefill_backend': 'triton'}
        with patch.object(legacy, 'install') as install:
            self.assertEqual(legacy.resolve_backend(cfg, 'triton'), 'triton')
            install.assert_called_once()

    def test_default_fp32_dense_and_foreign_geometry_remain_upstream(self):
        for section, field, value in [
            ('cache_config', 'mamba_ssm_cache_dtype', 'float32'),
            ('model_config.hf_text_config', 'model_type', 'qwen3_5_text'),
            ('model_config.hf_text_config', 'linear_num_value_heads', 48),
            ('model_config', 'quantization', 'compressed-tensors'),
            ('parallel_config', 'tensor_parallel_size', 1),
            ('parallel_config', 'enable_dbo', True),
        ]:
            cfg = config(); obj = cfg
            for part in section.split('.'):
                obj = getattr(obj, part)
            setattr(obj, field, value)
            with self.subTest(field=field, value=value), patch.object(legacy, 'install') as install:
                self.assertEqual(legacy.resolve_backend(cfg, 'auto'), 'auto')
                self.assertFalse(legacy.enabled())
                install.assert_not_called()
        with patch.dict(os.environ, {'VLLM_QWEN3_5_FP16_SSM': '0'}):
            self.assertFalse(legacy.model_eligible(config()))
        cfg = config(); cfg.speculative_config = object()
        self.assertFalse(legacy.model_eligible(cfg))
        with patch.dict(os.environ, {'VLLM_MACH_PROFILE': 'qwen35-2b-mxfp8-champion-v1'}):
            self.assertFalse(legacy.model_eligible(config()))

    def test_attention_requires_actual_native_instance_context(self):
        stock = object()
        kw = dict(num_seqs=32, heads=8, kv_heads=1, segments=16,
                  head_size=256, padded_head_size=256, bf16=True, use_fp8=False)
        legacy._ENABLED = True
        with patch.object(legacy._CONTEXT, 'attention', None, create=True):
            self.assertIs(legacy.attention(stock, **kw), stock)
        with patch.object(legacy._CONTEXT, 'attention', 'model.layers.3.self_attn.attn', create=True):
            for key, value in [('num_seqs', 48), ('heads', 16), ('kv_heads', 2),
                               ('segments', 64), ('head_size', 128), ('bf16', False),
                               ('use_fp8', True)]:
                self.assertIs(legacy.attention(stock, **{**kw, key: value}), stock)

    def test_qualified_grid_rows_use_same_attention_kernel(self):
        stock = object()
        target = object()
        module = 'vllm_mach.mxfp8.two_b._attention_reduction_kernel'
        with patch.object(legacy, '_ENABLED', True), patch.object(
                legacy._CONTEXT, 'attention', 'model.layers.3.self_attn.attn', create=True), patch.dict(
                    'sys.modules', {module: N(reduce_segments_explicit_old_lane=target)}):
            for rows in (1, 2, 4, 8, 16, 24, 32):
                with self.subTest(rows=rows):
                    result = legacy.attention(stock, num_seqs=rows, heads=8, kv_heads=1,
                        segments=16, head_size=256, padded_head_size=256, bf16=True, use_fp8=False)
                    self.assertIsNot(result, stock)
        self.assertEqual(legacy._PACKED_ROWS, (16, 24, 32))
        self.assertEqual(legacy._DECODE_ROWS, (1, 2, 4, 8, 16, 24, 32))


if __name__ == '__main__':
    unittest.main()
