"""Dense norm instance binding and guard ABI; no Torch/CUDA dependency."""
import contextlib
import copy
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

from vllm_mach.mxfp6.gdn import dense_norm, dense_prep, legacy
from vllm_mach.runtime import attention
from test_dense_prep_cpu import config


class Tensor:
    def __init__(self, shape, dtype, stride, device='cuda:0'):
        self.shape, self.dtype, self._stride = shape, dtype, stride
        self.ndim, self.device, self.is_cuda = len(shape), device, device.startswith('cuda')
    def stride(self):
        return self._stride


class DenseNormTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.bf16, self.fp32, self.fp16 = object(), object(), object()
        self.capturing = False
        self.torch = N(bfloat16=self.bf16, float32=self.fp32, float16=self.fp16,
                       cuda=N(is_current_stream_capturing=lambda: self.capturing))
        for p in [patch.dict(os.environ, {}, clear=True),
                  patch.object(attention, '_NATIVE_REGISTERED', True),
                  patch.object(dense_prep, '_sm120', return_value=True),
                  patch.object(dense_prep, '_ENABLED', True),
                  patch.object(dense_norm, '_INSTALLED', False),
                  patch.object(dense_norm, '_WARMED', set()),
                  patch.object(dense_norm, '_STATS', dict(prepared=[], calls={},
                    capture_calls={}, kernels={}, warmups={})),
                  patch.dict(sys.modules, {'torch': self.torch})]:
            self.stack.enter_context(p)
        dense_norm._CONTEXT.prefix = None
        self.calls = []
        calls = self.calls
        class Norm:
            def __init__(inner):
                inner.weight = Tensor((128,), self.bf16, (1,))
                inner.bias, inner.eps, inner.group_size = None, 1e-6, None
                inner.norm_before_gate, inner.activation = True, 'silu'
                inner._forward_method = inner.forward_cuda
            def forward_cuda(inner, x, z=None):
                calls.append(('forward', x, z, getattr(dense_norm._CONTEXT, 'prefix', None)))
                return x
        self.Norm = Norm
        self.stack.enter_context(patch.dict(sys.modules, {
            'vllm.model_executor.layers.layernorm': N(RMSNormGated=Norm)}))

    def layer(self, dtype=None):
        def direct(x, output_gate, out):
            self.calls.append(('direct', x, output_gate, out,
                               getattr(dense_norm._CONTEXT, 'prefix', None)))
            return None
        return N(prefix='model.layers.0.linear_attn', norm=self.Norm(),
                 _mach_gdn_prepared=True, vllm_config=config(), tp_size=2,
                 num_k_heads=16, num_v_heads=48,
                 get_state_dtype=lambda: (self.bf16, dtype or self.fp32),
                 _rms_norm_gated_cuda=direct, gdn_prefill_backend='flashinfer')

    def test_real_legacy_prepare_binds_dense_after_warmup_without_moe(self):
        for dtype in (self.fp32, self.fp16):
            layer = self.layer(dtype)
            with self.subTest(dtype=dtype), patch.object(legacy, '_ENABLED', False), patch.object(
                    dense_norm, '_install') as install, patch.object(
                    dense_norm, '_warmup', side_effect=lambda l:
                    self.assertFalse(getattr(l, '_mach_dense_norm', False))) as warm:
                legacy.prepare(layer)
                self.assertTrue(layer._mach_dense_norm)
                self.assertEqual(layer.gdn_prefill_backend, 'flashinfer')
                self.assertFalse(legacy.enabled())
                install.assert_called_once(); warm.assert_called_once_with(layer)
                bound = layer.norm._forward_method
                legacy.prepare(layer)
                self.assertIs(layer.norm._forward_method, bound)
                self.assertEqual(warm.call_count, 1)

    def test_unprepared_other_model_state_or_frontend_stays_unmodified(self):
        for field, value in [('_mach_gdn_prepared', False), ('tp_size', 1),
                             ('num_v_heads', 32), ('num_k_heads', 8)]:
            layer = self.layer(); original = layer.norm._forward_method
            setattr(layer, field, value)
            with self.subTest(field=field), patch.object(dense_norm, '_install') as install:
                dense_norm.prepare(layer); install.assert_not_called()
                self.assertIs(layer.norm._forward_method, original)
        for change in ('state', 'model', 'norm', 'native', 'disabled'):
            layer = self.layer(); original = layer.norm._forward_method
            if change == 'state': layer.get_state_dtype = lambda: (self.bf16, self.bf16)
            if change == 'model': layer.vllm_config.model_config.hf_text_config.hidden_size = 2048
            if change == 'norm': layer.norm.eps = 1e-5
            if change == 'native': layer.norm._forward_method = lambda x, z: x
            with self.subTest(change=change), patch.object(dense_norm, '_install') as install, patch.object(
                    dense_prep, '_ENABLED', change != 'disabled'):
                dense_norm.prepare(layer); install.assert_not_called()
                self.assertFalse(getattr(layer, '_mach_dense_norm', False))

    def test_forward_direct_kwargs_identity_scope_and_unknown_layout_fallback(self):
        layer = self.layer()
        with patch.object(dense_norm, '_install'), patch.object(dense_norm, '_warmup'):
            dense_norm.prepare(layer)
        x = Tensor((32,24,128), self.bf16, (3072,128,1))
        z = Tensor(x.shape, self.bf16, (8192,128,1))
        out = Tensor(x.shape, self.bf16, x.stride())
        self.assertIs(layer.norm._forward_method(x=x, z=z), x)
        self.assertEqual(self.calls[-1][-1], layer.prefix)
        self.assertIsNone(layer._rms_norm_gated_cuda(x=x, output_gate=z, out=out))
        self.assertEqual(self.calls[-1], ('direct', x, z, out, layer.prefix))
        self.assertIsNone(dense_norm._CONTEXT.prefix)
        for bad in [Tensor(x.shape,self.bf16,(6144,128,1)),
                    Tensor((32,16,128),self.bf16,(2048,128,1)),
                    Tensor(x.shape,self.fp16,x.stride()),
                    Tensor(x.shape,self.bf16,x.stride(),'cpu')]:
            with self.subTest(stride=bad.stride(),shape=bad.shape):
                self.assertIs(layer.norm._forward_method(bad, z), bad)
                self.assertIsNone(self.calls[-1][-1])
        badout = Tensor(x.shape,self.bf16,(8192,128,1))
        layer._rms_norm_gated_cuda(x,z,badout)
        self.assertIsNone(self.calls[-1][-1])

    def test_guard_exact_dispatch_fallback_capture_and_metadata_hash_once(self):
        downstream = []
        result = N(hash='one-real-kernel', metadata=N(num_warps=1,num_stages=3),asm={'ptx':'PTX'})
        def stock(*args,**kw): downstream.append(('stock',args,kw)); return 'stock'
        def target(*args,**kw): downstream.append(('target',args,kw)); return result
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'guard.py';path.write_text('qualified-guard')
            guard=N(__file__=str(path),layer_norm_fwd_kernel=N(run=stock))
            modules={'vllm.third_party.flash_linear_attention.ops':N(layernorm_guard=guard),
                     'vllm_mach.fp8._guard_kernel':N(layer_norm_fwd_kernel=N(run=target))}
            with patch.dict(sys.modules,modules),patch.object(dense_norm,'_GUARD_SHA256',hashlib.sha256(path.read_bytes()).hexdigest()):
                dense_norm._install();run=guard.layer_norm_fwd_kernel.run
                dense_norm._install();self.assertIs(guard.layer_norm_fwd_kernel.run,run)
                args=(object(),object(),object(),None,object(),None,object(),128,128,128,52008,128,1e-6)
                kw=dict(BLOCK_N=128,ROWS_PER_BLOCK=4,HAS_BIAS=False,HAS_Z=True,
                        IS_RMS_NORM=True,NORM_BEFORE_GATE=True,ACTIVATION='silu',num_warps=1)
                self.assertEqual(run(*args,**kw),'stock')
                dense_norm._CONTEXT.prefix='dense0'
                real_sha=hashlib.sha256
                with patch.object(dense_norm.hashlib,'sha256',wraps=real_sha) as hashes:
                    self.assertIs(run(*args,**kw),result)
                    self.capturing=True
                    self.assertIs(run(*args,**kw),result)
                    self.assertEqual(hashes.call_count,1)
                self.assertEqual(downstream[-1],('target',args,kw))
                for field,value in [('ROWS_PER_BLOCK',8),('BLOCK_N',256),('HAS_Z',False),('num_warps',2)]:
                    self.assertEqual(run(*args,**{**kw,field:value}),'stock')
                self.assertEqual(run(*args[:7],256,*args[8:],**kw),'stock')
                self.assertEqual(dense_norm._STATS['calls'],{'dense0:row4:m52008':2})
                self.assertEqual(dense_norm._STATS['capture_calls'],{'dense0:row4:m52008':1})


if __name__ == '__main__':
    unittest.main()
