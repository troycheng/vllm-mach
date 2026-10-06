"""AST scheduling boundaries and stream fork/join contracts, entirely CPU."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]/'src/vllm_mach/mxfp8'
def load(name):
    spec=importlib.util.spec_from_file_location('vllm_mach.mxfp8.'+name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def source(lid=1,qfirst=True,middle='        assert_size_stride(qout, (32, 12288), (12288, 1))\n'):
    q=f'        qout = torch.ops.vllm.mach_mx8_dual_qkvz_both.default(x, qw, qs, {lid})\n'
    b='        bout = torch.ops.mx8_ba_only.small_n.default(x, bw, bp)\n'
    return ('"""import triton\n@triton.jit\ndef untouched(): pass\n"""\n'
            'from __future__ import annotations\n'
            'import torch\n'
            'class Runner:\n'
            '    def call(self, args):\n'
            '        x, qw, qs, bw, bp = args\n'+
            (q+middle+b if qfirst else b+middle+q)+
            '        return qout, bout\n')

class ProjectionTransformContracts(unittest.TestCase):
    def setUp(self):self.transform=load('projection_transform')
    def test_line_preserving_rewrite_and_future_import(self):
        original=source()
        changed,audit=self.transform.transform_source(original)
        self.assertIn('from vllm_mach.mxfp8.projection_parallel import dispatch',changed)
        self.assertEqual(ast.get_docstring(ast.parse(original)),ast.get_docstring(ast.parse(changed)))
        self.assertEqual(audit['pairs'][0]['layer_id'],1)
        self.assertEqual(len(audit['changed_call_lines']),2)
        before=original.splitlines();after=changed.splitlines();after.pop(audit['import_before_line']-1)
        for lineno,(left,right) in enumerate(zip(before,after),1):
            if lineno not in audit['changed_call_lines']:self.assertEqual(left,right)
        ast.parse(changed)
    def test_first_operation_order_and_layer0_preservation(self):
        changed,audit=self.transform.transform_source(source(qfirst=False,middle=''))
        self.assertIn('1, bw, bp, False)',changed)
        self.assertEqual(audit['pairs'][0]['first'],'B')
        original=source(lid=0,middle='        arbitrary_original_gpu_work()\n')
        changed,audit=self.transform.transform_source(original)
        self.assertEqual(changed,original);self.assertEqual(audit['skipped_layer_ids'],[0])
    def test_reject_side_effects_deletions_reassigned_input_and_hidden_calls(self):
        for middle in ('        arbitrary_gpu_work()\n','        del x\n','        x = qw\n',
                       '        assert_size_stride(arbitrary_gpu_work(), (), ())\n',
                       '        bw = qs\n'):
            with self.assertRaises(ValueError):self.transform.transform_source(source(middle=middle))
        original=source().replace('default(x, bw, bp)','default(other, bw, bp)')
        with self.assertRaises(ValueError):self.transform.transform_source(original)
        original=source().replace('qout = torch.ops','x = torch.ops')
        with self.assertRaises(ValueError):self.transform.transform_source(original)
    def test_reject_unpaired_outside_call_target_and_helper_collision(self):
        for original in (source().replace('        bout = torch.ops.mx8_ba_only.small_n.default(x, bw, bp)\n',''),
                         source()+'outside = torch.ops.mx8_ba_only.small_n.default(x, bw, bp)\n',
                         source()+'other = f(torch.ops.mx8_ba_only.small_n.default(x, bw, bp))\n',
                         source()+'_qkvz_ba_pair = 1\n'):
            with self.assertRaises(ValueError):self.transform.transform_source(original)
    def test_deferred_alignment_check_remains_at_original_line(self):
        original=source(middle='        bw = copy_if_misaligned(bw)\n        bp = copy_if_misaligned(bp)\n')
        changed,audit=self.transform.transform_source(original)
        self.assertEqual(audit['pairs'][0]['late_alignment_weights'],['bw','bp'])
        self.assertIn('        bw = copy_if_misaligned(bw)\n',changed)

    def test_closed_stream_dependencies_and_original_serial_order(self):
        runtime=load('projection_parallel')
        events=[]
        class Stream:
            def __init__(self,name):self.name=name
            def wait_stream(self,other):events.append((self.name,'wait',other.name))
        origin=Stream('origin');aux=Stream('aux')
        class Context:
            def __enter__(self):events.append('aux_enter')
            def __exit__(self,*args):events.append('aux_exit')
        torch=types.ModuleType('torch');torch.bfloat16='bf16'
        torch.cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False,
            current_stream=lambda device:origin,stream=lambda stream:Context())
        def qop(*args):events.append('qkvz');return 'Q'
        def bop(*args):events.append('ba');return 'B'
        torch.ops=types.SimpleNamespace(vllm=types.SimpleNamespace(mach_mx8_dual_qkvz_both=types.SimpleNamespace(default=qop)),
            mx8_ba_only=types.SimpleNamespace(small_n=types.SimpleNamespace(default=bop)))
        tensor=lambda ptr:types.SimpleNamespace(data_ptr=lambda:ptr)
        weights=[tensor(p) for p in (16,32,48,64)]
        x=types.SimpleNamespace(dtype='bf16',ndim=2,shape=(32,2560),device='cuda',is_contiguous=lambda:True)
        runtime._READY=True;runtime._AUX=aux
        runtime._PAIRS={i:{'pointers':(16,32,48,64)} for i in (0,1)}
        with patch.dict(sys.modules,{'torch':torch}):
            self.assertEqual(runtime.dispatch(x,weights[0],weights[1],1,weights[2],weights[3],True),('Q','B'))
            self.assertEqual(events,[('aux','wait','origin'),'aux_enter','ba','aux_exit','qkvz',('origin','wait','aux')])
            events.clear();runtime.dispatch(x,weights[0],weights[1],0,weights[2],weights[3],False)
            self.assertEqual(events,['ba','qkvz'])
            events.clear();x.shape=(16,2560)
            runtime.dispatch(x,weights[0],weights[1],1,weights[2],weights[3],True)
            self.assertEqual(events,['qkvz','ba'])
            with self.assertRaisesRegex(RuntimeError,'identity'):
                runtime.dispatch(x,tensor(80),weights[1],1,weights[2],weights[3],True)

if __name__=='__main__':unittest.main()
