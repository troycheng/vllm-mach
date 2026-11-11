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

    def test_prefill_wrapper_accepts_real_qwen_keyword_and_mixed_calls(self):
        try:
            import ast
            import functools
            import inspect
            from pathlib import Path
            import torch
            import triton
            from torch._subclasses.fake_tensor import FakeTensorMode
            from vllm.third_party.flash_linear_attention.ops import fused_gdn_prefill_post_conv as upstream
            from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
            from vllm_mach import _gdn_prep_kernel as shared
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        # Compile the actual nested product wrapper, retaining its real Python
        # argument binder. wraps() otherwise makes inspect.signature hide drift.
        tree = ast.parse(inspect.getsource(legacy.install))
        wrapper = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'prep')
        namespace = dict(vars(legacy), original_prep=upstream.fused_post_conv_prep,
                         torch=torch, _ENABLED=True, _STATS={'prep_calls': 0})
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<actual-native-prep-wrapper>', 'exec'), namespace)
        prep = namespace['prep']
        abi = lambda fn: [(n, p.kind, p.default) for n, p in inspect.signature(fn, follow_wrapped=False).parameters.items()]
        self.assertEqual(abi(prep), abi(upstream.fused_post_conv_prep))
        self.assertEqual(abi(prep), abi(shared.fused_post_conv_prep))
        prepare_tree = ast.parse(inspect.getsource(legacy.prepare))
        decode = next(n for n in prepare_tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'decode_norm')
        self.assertEqual([p.arg for p in decode.args.args], list(inspect.signature(qwen.QwenGatedDeltaNetAttention._rms_norm_gated_cuda).parameters)[1:])
        sentinel = object()
        fallback = dict(vars(legacy), torch=torch, layer=N(get_state_dtype=lambda: (torch.bfloat16, torch.float32)), original_decode=lambda x, output_gate, out: sentinel)
        exec(compile(ast.Module(body=[decode], type_ignores=[]), '<actual-native-decode-norm-wrapper>', 'exec'), fallback)
        self.assertIs(fallback['decode_norm'](x=object(), output_gate=object(), out=object()), sentinel)
        calls = []
        class DeviceLaunch:
            def __init__(self, name): self.name = name
            def __getitem__(self, grid):
                def run(*args, **kwargs): calls.append((self.name, grid, args, kwargs))
                return run
        with FakeTensorMode(), patch.object(upstream, 'triton', triton), patch.object(shared, 'triton', triton), patch.object(upstream, '_fused_post_conv_kernel', DeviceLaunch('upstream')), patch.object(shared, '_fused_post_conv_kernel', DeviceLaunch('qualified')):
            conv = torch.empty(64, 4096, dtype=torch.bfloat16, device='cuda:0')
            a = torch.empty(64, 16, dtype=torch.bfloat16, device='cuda:0')
            values = dict(conv_output=conv, a=a, b=a, A_log=torch.empty(16, device='cuda:0'), dt_bias=torch.empty(16, dtype=torch.bfloat16, device='cuda:0'), num_k_heads=8, head_k_dim=128, head_v_dim=128, apply_l2norm=True, output_g_exp=False)
            # Both real qwen warmup and runtime callsites are keyword calls.
            sites = [n for n in ast.walk(ast.parse(Path(qwen.__file__).read_text())) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fused_post_conv_prep']
            self.assertEqual(len(sites), 2)
            for call in sites:
                self.assertFalse(call.args)
                self.assertEqual({k.arg for k in call.keywords}, set(values))
                self.assertEqual(len(prep(**{k.arg: values[k.arg] for k in call.keywords})), 5)
            self.assertEqual(len(prep(*values.values())), 5)
            first = list(values)[:5]
            self.assertEqual(len(prep(*(values[k] for k in first), **{k: v for k, v in values.items() if k not in first})), 5)
            # An unmatched frontend keeps the actual upstream body/allocation.
            self.assertEqual(len(prep(**{**values, 'apply_l2norm': False})), 5)
        self.assertEqual([x[0] for x in calls], ['qualified'] * 4 + ['upstream'])
        self.assertEqual(namespace['_STATS']['prep_calls'], 4)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
