"""Block M4 packed scope and the exact four-expression source replacement."""
import ast
import os
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch

from vllm_mach.fp8 import packed_fp32 as p

SOURCE = '''def kernel(b_q,b_k,b_h,BV):
    untouched = b_h + 1.0
    q = tl.sum(b_q*b_q)
    k = tl.sum(b_k*b_k)
    v = tl.sum(b_h*b_k[None,:],1)
    o = tl.sum(b_h*b_q[None,:],1)
    tl.store(ptr, untouched)
'''

class PackedScope(unittest.TestCase):
    def test_four_reductions_preserve_other_arithmetic_and_reject_drift(self):
        changed = ast.parse(p.rewrite(SOURCE)).body[0]
        original = ast.parse(SOURCE).body[0]
        self.assertEqual(ast.dump(changed.body[0]), ast.dump(original.body[0]))
        self.assertEqual(ast.dump(changed.body[-1]), ast.dump(original.body[-1]))
        calls = [n.func.id for n in ast.walk(changed)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertEqual(calls.count('_old_square_128'), 2)
        self.assertEqual(calls.count('_old_dot_128'), 2)
        for text in (SOURCE.replace('tl.sum(b_q*b_q)', 'tl.sum(b_q)'),
                     SOURCE.replace('    q = ', '    duplicate = tl.sum(b_q*b_q)\n    q = ')):
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                p.rewrite(text)

    def test_only_qualified_M4_FP32_layout_family(self):
        bound = dict(H=16, HV=32, K=128, V=128, BK=128, BV=32,
                     h0=N(dtype='float32'), ht=N(dtype='float32'),
                     mixed_qkv=N(shape=(4, 8192)), SPLIT_BATCH_HEAD_GRID=False,
                     USE_QK_L2NORM_IN_KERNEL=True,SOFTPLUS_THRESHOLD=20.0,scale=128**-0.5)
        self.assertTrue(p.admitted(bound, 'float32'))
        for key, value in [('HV',16), ('H',8), ('BK',64), ('BV',64),
                           ('SPLIT_BATCH_HEAD_GRID',True), ('USE_QK_L2NORM_IN_KERNEL',False),
                           ('scale',0.5),('SOFTPLUS_THRESHOLD',10.0), ('h0',N(dtype='bf16')),
                           ('ht',N(dtype='fp16')), ('mixed_qkv',N(shape=(32,8192)))]:
            self.assertFalse(p.admitted({**bound,key:value}, 'float32'),key)
        with patch.dict(os.environ, {'VLLM_MACH_PROFILE':'ordinary'}):
            self.assertFalse(p.install())

    def test_proxy_keeps_launch_ABI_ownership_fallback_and_binding_identity(self):
        import hashlib
        import sys
        import tempfile
        from pathlib import Path
        import types
        def frontend(mixed_qkv,h0,ht,H,HV,K,V,BK,BV,SPLIT_BATCH_HEAD_GRID,
                     stride_mixed_qkv_tok,stride_init_state_token,stride_final_state_token,
                     USE_QK_L2NORM_IN_KERNEL,SOFTPLUS_THRESHOLD,scale):
            pass
        events=[]
        class Kernel:
            src='pinned source'
            fn=staticmethod(frontend)
            def __init__(self,label):self.label=label
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    events.append((self.label,grid,a,kw))
                    return N(hash='compiled',metadata=N(num_warps=1,num_stages=3,shared=128))
                return launch
        original=Kernel('original');candidate=Kernel('candidate')
        module=types.ModuleType('vllm.third_party.flash_linear_attention.ops.fused_recurrent')
        ops=types.ModuleType('vllm.third_party.flash_linear_attention.ops')
        ops.fused_recurrent=module
        setattr(module,p._NAME,original)
        torch=N(float32='fp32',cuda=N(is_current_stream_capturing=lambda:True))
        pool=N(dtype='fp32',stride=lambda:(540672,16384,128,1),data_ptr=lambda:123)
        values=dict(mixed_qkv=N(shape=(4,8192)),h0=pool,ht=pool,H=16,HV=32,K=128,V=128,BK=128,BV=32,
                    SPLIT_BATCH_HEAD_GRID=False,stride_mixed_qkv_tok=12288,
                    stride_init_state_token=540672,stride_final_state_token=540672,
                    USE_QK_L2NORM_IN_KERNEL=True,SOFTPLUS_THRESHOLD=20.0,scale=128**-0.5,
                    num_warps=1,num_stages=3)
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'kernel.py';file.write_text('actual pinned file');module.__file__=str(file)
            with patch.dict(sys.modules,{'torch':torch,ops.__name__:ops,module.__name__:module}), \
                 patch.dict(os.environ,{'VLLM_MACH_PROFILE':p.NAME}), \
                 patch.multiple(p,_STATE=None,_ORIGINAL=None,
                     _SOURCE_SHA=hashlib.sha256(file.read_bytes()).hexdigest(),
                     _JIT_SHA=hashlib.sha256(original.src.encode()).hexdigest()), \
                 patch.object(p,'_kernel',return_value=candidate):
                self.assertTrue(p.install());proxy=getattr(module,p._NAME)
                self.assertFalse(p.install())
                grid=lambda meta:(4,128)
                proxy[grid](**values)
                self.assertIs(events[-1][1],grid)
                self.assertEqual(events[-1][3],values)
                self.assertEqual(events[-1][0],'candidate')
                self.assertEqual(p.receipt()['capturing_calls'],1)
                proxy[grid](**{**values,'mixed_qkv':N(shape=(32,8192))})
                self.assertEqual(events[-1][0],'original')
                with self.assertRaisesRegex(RuntimeError,'layout/config'):
                    proxy[grid](**{**values,'num_warps':2})
                setattr(module,p._NAME,original)
                with self.assertRaisesRegex(RuntimeError,'binding changed'):p.receipt()
