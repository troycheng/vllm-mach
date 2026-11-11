"""Dense prep admission and forwarding; no CUDA or Torch import is needed."""
import copy
import os
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch

from vllm_mach.mxfp6.gdn import dense_prep, legacy
from vllm_mach.runtime import attention


def config():
    weight = dict(dtype='fp6_e3m2', group_size=32, qscheme='per_group',
                  scale_format='e8m0', is_dynamic=False, symmetric=True)
    activation = dict(dtype='fp8_e4m3', group_size=32, qscheme='per_group',
                      scale_format='e8m0', is_dynamic=True, symmetric=True)
    text = N(model_type='qwen3_5_text', hidden_size=5120, num_hidden_layers=64,
             linear_num_key_heads=16, linear_num_value_heads=48,
             linear_key_head_dim=128, linear_value_head_dim=128)
    return N(model_config=N(dtype='torch.bfloat16', quantization='quark', hf_text_config=text,
                           model_arch_config=N(quantization_config={'global_quant_config':
                           {'weight': weight, 'input_tensors': activation}})),
             parallel_config=N(tensor_parallel_size=2, pipeline_parallel_size=1,
                               decode_context_parallel_size=1, enable_dbo=False),
             cache_config=N(cache_dtype='auto', mamba_ssm_cache_dtype='float32'),
             speculative_config=None, additional_config={})


class DensePrepAdmission(unittest.TestCase):
    def setUp(self):
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch.object(attention, '_NATIVE_REGISTERED', True))
        self.stack.enter_context(patch.object(dense_prep, '_ENABLED', False))
        self.stack.enter_context(patch.object(dense_prep, '_INSTALLED', False))
        self.stack.enter_context(patch.object(dense_prep, '_STATS',
                                             dict(calls=0, capture_calls=0, rows={})))
        self.stack.enter_context(patch.object(dense_prep, '_sm120', return_value=True))

    def test_state_independent_prep_does_not_enable_any_moe_bundle(self):
        for dtype in ('float32', 'float16'):
            for requested in ('auto', 'flashinfer', 'triton'):
                cfg = config()
                cfg.cache_config.mamba_ssm_cache_dtype = dtype
                with self.subTest(dtype=dtype, requested=requested), patch.object(
                        dense_prep, 'install') as install, patch.object(
                        legacy, 'install') as moe_install, patch.object(legacy, '_ENABLED', False):
                    self.assertEqual(legacy.resolve_backend(cfg, requested), requested)
                    install.assert_called_once()
                    moe_install.assert_not_called()
                    self.assertFalse(legacy.enabled())

    def test_other_models_modes_quantization_and_hardware_keep_upstream(self):
        cases = [('model_config.hf_text_config.model_type', 'qwen3_5_moe_text'),
                 ('model_config.hf_text_config.hidden_size', 2048),
                 ('model_config.hf_text_config.linear_num_value_heads', 32),
                 ('model_config.quantization', 'fp8'),
                 ('parallel_config.tensor_parallel_size', 1),
                 ('parallel_config.enable_dbo', True),
                 ('cache_config.mamba_ssm_cache_dtype', 'bfloat16'),
                 ('cache_config.mamba_ssm_cache_dtype', 'auto'),
                 ('speculative_config', object())]
        for field, value in cases:
            cfg = copy.deepcopy(config())
            owner = cfg
            names = field.split('.')
            for name in names[:-1]:
                owner = getattr(owner, name)
            setattr(owner, names[-1], value)
            with self.subTest(field=field), patch.object(dense_prep, 'install') as install:
                dense_prep.configure(cfg)
                self.assertFalse(dense_prep._ENABLED)
                install.assert_not_called()
        with patch.object(dense_prep, '_sm120', return_value=False), patch.object(
                dense_prep, 'install') as install:
            dense_prep.configure(config())
            install.assert_not_called()
        with patch.dict(os.environ, {'VLLM_MACH_PROFILE': 'qwen35-2b-mxfp8-champion-v1'}):
            self.assertFalse(dense_prep.model_eligible(config()))
        with patch.object(attention, '_NATIVE_REGISTERED', False):
            self.assertFalse(dense_prep.model_eligible(config()))

    def test_exact_wrapper_abi_fallback_and_actual_call_counters(self):
        bf16, fp32 = object(), object()
        torch = N(bfloat16=bf16, float32=fp32,
                  cuda=N(is_current_stream_capturing=lambda: False))
        class Tensor:
            def __init__(self, shape, dtype, stride):
                self.shape, self.dtype, self._stride = shape, dtype, stride
                self.ndim, self.is_cuda, self.device = len(shape), True, 'cuda:0'
            def stride(self):
                return self._stride
        calls = []
        def stock(*args):
            calls.append(('stock', args))
            return 'stock'
        def target(*args):
            calls.append(('target', args))
            return 'target'
        qwen = N(fused_post_conv_prep=stock)
        modules = {'torch': torch, 'vllm_mach._gdn_prep_kernel':
                   N(fused_post_conv_prep=target)}
        args = dict(conv_output=Tensor((2167, 5120), bf16, (5120, 1)),
                    a=Tensor((2167, 24), bf16, (24, 1)),
                    b=Tensor((2167, 24), bf16, (24, 1)),
                    A_log=Tensor((24,), fp32, (1,)),
                    dt_bias=Tensor((24,), bf16, (1,)),
                    num_k_heads=8, head_k_dim=128, head_v_dim=128,
                    apply_l2norm=True, output_g_exp=False)
        with patch.dict('sys.modules', modules), patch.object(
                dense_prep.importlib, 'import_module', return_value=qwen):
            dense_prep._ENABLED = True
            dense_prep.install()
            bound = qwen.fused_post_conv_prep
            dense_prep.install()
            self.assertIs(qwen.fused_post_conv_prep, bound)
            self.assertEqual(bound(**args), 'target')
            self.assertEqual(bound(*args.values()), 'target')
            self.assertEqual(bound(args['conv_output'], **{k: v for k, v in args.items()
                                                         if k != 'conv_output'}), 'target')
            for field, value in [('num_k_heads', 16), ('apply_l2norm', False),
                                 ('output_g_exp', True),
                                 ('a', Tensor((2167, 24), bf16, (48, 1))),
                                 ('conv_output', Tensor((2167, 4096), bf16, (4096, 1))),
                                 ('A_log', Tensor((16,), fp32, (1,)))]:
                with self.subTest(field=field):
                    self.assertEqual(bound(**{**args, field: value}), 'stock')
            dense_prep._ENABLED = False
            self.assertEqual(bound(**args), 'stock')
        self.assertEqual(dense_prep._STATS['calls'], 3)
        self.assertEqual(dense_prep._STATS['capture_calls'], 0)
        self.assertEqual(dense_prep._STATS['rows'], {2167: 3})
        self.assertTrue(all(call[1] == tuple(args.values()) for call in calls[:3]))


if __name__ == '__main__':
    unittest.main()
