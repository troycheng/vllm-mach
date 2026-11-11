"""Dense FP32 NT16 instance scope, lifecycle and unchanged launch forwarding."""
import ast
import contextlib
import copy
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

from vllm_mach.mxfp6.gdn import dense_attention, dense_prefill, dense_prep, legacy
from vllm_mach.runtime import attention
from test_dense_prep_cpu import config


class Tensor:
    def __init__(self, shape, dtype, device='cuda:0', contiguous=True):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.is_cuda, self.contiguous = device.startswith('cuda'), contiguous
    def numel(self):
        value = 1
        for n in self.shape:
            value *= n
        return value
    def is_contiguous(self):
        return self.contiguous


class DenseAttentionTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.bf16, self.fp32, self.i32 = object(), object(), object()
        self.capturing = False
        self.launches = []
        self.torch = N(bfloat16=self.bf16, float32=self.fp32, int32=self.i32,
                       cuda=N(is_current_stream_capturing=lambda: self.capturing))
        self.torch.empty = lambda shape, *, dtype, device: Tensor(shape, dtype, device)
        self.torch.zeros = self.torch.empty
        self.torch.ones_like = lambda x: Tensor(x.shape, x.dtype, x.device)
        self.torch.full = lambda shape, value, *, dtype, device: Tensor(shape, dtype, device)
        self.torch.arange = lambda n, *, dtype, device: Tensor((n,), dtype, device)
        source = Path(dense_attention.__file__).parents[2] / 'mxfp8/two_b/_attention_reduction_kernel.py'
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef))
        names = [p.arg for p in fn.args.args]
        calls = self.launches
        class Kernel:
            arg_names = names
            def __init__(self, kind):
                self.kind = kind
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    values = {**dict(zip(names, args)), **kwargs}
                    calls.append((self.kind, grid, args, kwargs))
                    pitch = values.get('block_table_stride', 8)
                    key = 1 if pitch == 1 else 16 if pitch % 16 == 0 else 8
                    return N(hash='compiled-' + str(key), metadata=N(num_warps=4, num_stages=3), asm={'ptx': str(key)})
                return run
        self.target, self.stock = Kernel('target'), Kernel('stock')
        for p in [patch.dict(os.environ, {}, clear=True),
                  patch.dict(sys.modules, {'torch': self.torch}),
                  patch.object(attention, '_NATIVE_REGISTERED', True),
                  patch.object(dense_prep, '_sm120', return_value=True),
                  patch.object(dense_attention, '_ENABLED', False),
                  patch.object(dense_attention, '_WARMED', set()),
                  patch.object(dense_attention, '_STATS', dict(prepared=[], calls={}, capture_calls={}, kernels={}, warmups={})),
                  patch.object(dense_attention, '_target', return_value=self.target),
                  patch.object(legacy, '_ENABLED', False)]:
            self.stack.enter_context(p)
        dense_attention._CONTEXT.prefix = None
        outer = self
        class Impl:
            num_heads, num_kv_heads, head_size = 12, 2, 256
            def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None):
                chosen = legacy.attention(outer.stock, num_seqs=attn_metadata.seq_lens.shape[0],
                    heads=self.num_heads, kv_heads=self.num_kv_heads, segments=16,
                    head_size=256, padded_head_size=256, bf16=query.dtype == outer.bf16, use_fp8=False)
                chosen[(32, 12)](**outer.values(output))
                return output
        self.Impl = Impl
        self.stack.enter_context(patch.dict(sys.modules, {
            'vllm.v1.attention.backends.triton_attn': N(TritonAttentionImpl=Impl)}))

    def cfg(self):
        cfg = config()
        cfg.model_config.hf_text_config.num_attention_heads = 24
        cfg.model_config.hf_text_config.num_key_value_heads = 4
        cfg.model_config.hf_text_config.head_dim = 256
        return cfg

    def layer(self, index=3):
        return N(layer_name=f'model.layers.{index}.self_attn.attn', impl=self.Impl(),
                 _q_scale=Tensor((), self.fp32))

    def values(self, output=None):
        return dict(output_ptr=output or Tensor((32, 12, 256), self.bf16),
                    segm_output_ptr=Tensor((32, 12, 16, 256), self.fp32),
                    segm_max_ptr=Tensor((32, 12, 16), self.fp32),
                    segm_expsum_ptr=Tensor((32, 12, 16), self.fp32),
                    seq_lens_ptr=Tensor((32,), self.i32), num_seqs=32, num_query_heads=12,
                    out_scale_inv=1., output_stride_0=3072, output_stride_1=256,
                    block_table_stride=8, TILE_SIZE=16, HEAD_SIZE=256, HEAD_SIZE_PADDED=256,
                    query_start_len_ptr=Tensor((33,), self.i32), BLOCK_Q=2,
                    NUM_SEGMENTS_PER_SEQ=16, USE_FP8=False)

    def prepare(self):
        dense_attention.configure(self.cfg())
        layer = self.layer()
        model = N(modules=lambda: [layer])
        legacy.prepare_attention(model)
        return layer, model

    def test_real_legacy_resolver_configures_dense_without_moe_or_backend_override(self):
        for requested in ('auto', 'flashinfer', 'triton'):
            cfg = self.cfg(); cfg.additional_config = {'gdn_prefill_backend': requested}
            with self.subTest(requested=requested), patch.object(dense_prep, 'install'), patch.object(
                    dense_prefill, 'install'), patch.object(legacy, 'install') as moe:
                self.assertEqual(legacy.resolve_backend(cfg, requested), requested)
                self.assertTrue(dense_attention._ENABLED)
                self.assertFalse(legacy.enabled()); moe.assert_not_called()

    def test_fp16_foreign_geometry_model_and_hardware_remain_stock(self):
        cases = [('cache_config.mamba_ssm_cache_dtype', 'float16'),
                 ('model_config.hf_text_config.num_key_value_heads', 2),
                 ('model_config.hf_text_config.hidden_size', 2048),
                 ('parallel_config.tensor_parallel_size', 1)]
        for field, value in cases:
            cfg = copy.deepcopy(self.cfg()); owner = cfg
            parts = field.split('.')
            for part in parts[:-1]: owner = getattr(owner, part)
            setattr(owner, parts[-1], value)
            with self.subTest(field=field):
                dense_attention.configure(cfg); self.assertFalse(dense_attention._ENABLED)
        with patch.object(dense_prep, '_sm120', return_value=False):
            dense_attention.configure(self.cfg()); self.assertFalse(dense_attention._ENABLED)

    def test_prepare_warms_three_pitch_keys_before_binding_without_cache(self):
        layer, model = self.prepare()
        self.assertFalse(hasattr(layer, 'kv_cache'))
        self.assertTrue(layer.impl._mach_dense_fp32_attention)
        self.assertEqual([r[3]['block_table_stride'] for r in self.launches], [1, 8, 16])
        self.assertFalse(dense_attention._STATS['calls'])
        self.assertFalse(dense_attention._STATS['capture_calls'])
        self.assertEqual(dense_attention._STATS['warmups'], {'cuda:0': 3})
        bound = layer.impl.forward
        legacy.prepare_attention(model)
        self.assertIs(layer.impl.forward, bound)
        self.assertEqual(len(self.launches), 3)

    def test_forward_pos_keyword_mixed_restores_context_and_original_output(self):
        layer, _ = self.prepare(); self.launches.clear(); self.capturing = True
        values = dict(layer=layer, query=Tensor((32, 3072), self.bf16), key=object(),
                      value=object(), kv_cache=object(),
                      attn_metadata=N(max_query_len=1, num_actual_tokens=32,
                                      seq_lens=Tensor((32,), self.i32)),
                      output=Tensor((32, 12, 256), self.bf16))
        forward = layer.impl.forward
        # Metadata/PTX identity is cached by warmup, never rehashed per step.
        with patch.object(dense_attention.hashlib, 'sha256', side_effect=AssertionError('hot-path hashing')):
            self.assertIs(forward(**values), values['output'])
            self.assertIs(forward(*values.values()), values['output'])
            self.assertIs(forward(layer, **{k: v for k, v in values.items() if k != 'layer'}), values['output'])
        self.assertEqual([r[0] for r in self.launches], ['target'] * 3)
        self.assertIsNone(dense_attention._CONTEXT.prefix)
        self.assertEqual(dense_attention._STATS['calls'], {layer.layer_name: 3})
        self.assertEqual(dense_attention._STATS['capture_calls'], {layer.layer_name: 3})
        values['attn_metadata'].num_actual_tokens = 4
        forward(**values)
        self.assertEqual(self.launches[-1][0], 'stock')

    def test_exact_launch_falls_back_unchanged_outside_qualified_scope(self):
        layer, _ = self.prepare(); self.launches.clear()
        dense_attention._CONTEXT.prefix = layer.layer_name
        selected = legacy.attention(self.stock, num_seqs=32, heads=12, kv_heads=2,
            segments=16, head_size=256, padded_head_size=256, bf16=True, use_fp8=False)
        values = self.values()
        for pitch in (1, 4, 8, 16, 32):
            selected[(32, 12)](**{**values, 'block_table_stride': pitch})
            self.assertEqual(self.launches[-1][0], 'target')
        capacity = {**values, 'segm_output_ptr': Tensor((64, 12, 16, 256), self.fp32),
                    'segm_max_ptr': Tensor((64, 12, 16), self.fp32),
                    'segm_expsum_ptr': Tensor((64, 12, 16), self.fp32)}
        selected[(32, 12)](**capacity)
        self.assertEqual(self.launches[-1], ('target', (32, 12), (), capacity))
        for key, value in [('num_query_heads', 8), ('BLOCK_Q', 4),
                           ('USE_FP8', True), ('num_warps', 8),
                           ('segm_max_ptr', Tensor((32, 12, 16), self.fp32, contiguous=False)),
                           ('segm_output_ptr', Tensor((31, 12, 16, 256), self.fp32)),
                           ('block_table_stride', 0)]:
            changed = {**values, key: value}
            selected[(32, 12)](**changed)
            self.assertEqual(self.launches[-1], ('stock', (32, 12), (), changed))
        self.assertIs(legacy.attention(self.stock, num_seqs=4, heads=12, kv_heads=2,
            segments=16, head_size=256, padded_head_size=256, bf16=True, use_fp8=False), self.stock)


if __name__ == '__main__':
    unittest.main()
