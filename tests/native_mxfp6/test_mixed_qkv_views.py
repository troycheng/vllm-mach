"""New installation boundaries; arithmetic/state proof is the qualified component."""
import ast
import copy
import hashlib
import importlib.metadata as metadata
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_mach.mxfp6 import gdn
from vllm_mach.mxfp6.gdn import legacy, mixed_qkv_views as views


def body_sha(node):
    return hashlib.sha256(ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()


class MixedQKVViewsContracts(unittest.TestCase):
    def test_qualified_helper_bodies_are_literal(self):
        tree=ast.parse(Path(views.__file__).read_text())
        nodes={n.name:n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)}
        expected=BODY_SHA
        self.assertEqual({name:body_sha(nodes[name])for name in expected},expected)

    def test_complete_core_inverse_includes_prefill_and_copy(self):
        try:path=Path(metadata.distribution('vllm').locate_file('vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'))
        except metadata.PackageNotFoundError:self.skipTest('actual installed vLLM source unavailable')
        self.assertEqual(views.digest(path),views.QWEN_SHA)
        compiled=compile(path.read_text(),str(path),'exec',dont_inherit=True)
        cls=next(c for c in compiled.co_consts if isinstance(c,types.CodeType)and c.co_name=='QwenGatedDeltaNetAttention')
        code=next(c for c in cls.co_consts if isinstance(c,types.CodeType)and c.co_name=='_forward_core')
        original=types.FunctionType(code,{'__file__':str(path)})
        node=views.source_node(original);_,proof=views.transform(node)
        self.assertTrue(proof['full_function_inverse_AST_equal'])
        self.assertTrue(proof['prefill_and_final_copy_unchanged'])

    def test_N1_and_rejection_do_not_enter_private_update(self):
        helper=object.__new__(views.Helpers)
        helper.can_view=lambda *args:(_ for _ in ()).throw(AssertionError('N1 reached guard'))
        md=types.SimpleNamespace(num_decode_tokens=1)
        self.assertEqual(helper.direct(object(),md,None,None,None,None,None,None),(False,None))
        helper.can_view=lambda *args:False;md.num_decode_tokens=2
        self.assertEqual(helper.direct(object(),md,None,None,None,None,None,None),(False,None))

    def test_admitted_call_once_and_exception_never_falls_back(self):
        helper=object.__new__(views.Helpers);values={'token':object()};events=[];sentinel=object();failure=ValueError('original private exception')
        helper.admitted_views=lambda *args:values;helper.stock=object()
        def admitted(*args,**kwargs):events.append(kwargs);return sentinel,None
        helper.admitted=admitted
        self.assertEqual(helper.direct(object(),object(),None,None,None,None,None,None),(True,sentinel));self.assertEqual(events,[values])
        def raises(*args,**kwargs):events.append(kwargs);raise failure
        helper.admitted=raises
        with self.assertRaises(ValueError)as caught:helper.direct(object(),object(),None,None,None,None,None,None)
        self.assertIs(caught.exception,failure);self.assertEqual(len(events),2)

    def test_factory_preserves_globals_future_defaults_and_single_closure(self):
        namespace={};source='from __future__ import annotations\ndef _forward_core(self, value: Missing = 7) -> Missing:\n return value\n'
        exec(compile(source,'future_fixture','exec',dont_inherit=True),namespace)
        original=namespace['_forward_core'];node=ast.parse(source).body[1]
        node.body=ast.parse('return _qkv_helper(self, value)').body
        sentinel=object();helper=lambda *args:sentinel
        candidate=views.bind(node,original,helper)
        self.assertIs(candidate.__globals__,original.__globals__);self.assertIs(candidate.__defaults__,original.__defaults__)
        self.assertIs(candidate.__kwdefaults__,original.__kwdefaults__);self.assertIs(candidate.__annotations__,original.__annotations__)
        self.assertEqual(candidate.__code__.co_freevars,('_qkv_helper',));self.assertIs(candidate.__closure__[0].cell_contents,helper)
        self.assertIs(candidate(object()),sentinel)

    def test_prepare_nonselected_does_not_import_runtime(self):
        with patch('builtins.__import__',side_effect=AssertionError('runtime import')):
            self.assertFalse(views.prepare(object()))
            self.assertFalse(views.prepare(types.SimpleNamespace(_mach_native_mixed_fp16=True)))

    def test_prepare_instance_ownership_idempotence_and_code_drift(self):
        qwen=types.ModuleType('qwen_fixture');qwen.__file__='qwen_fixture';exec('def _forward_core(self): return None',qwen.__dict__)
        original=qwen._forward_core
        Layer=type('QwenGatedDeltaNetAttention',(),{'_forward_core':original,'get_state_dtype':lambda self:(None,'fp16')})
        qwen.QwenGatedDeltaNetAttention=Layer
        native=types.ModuleType('mixed_fixture');native.__file__='mixed_fixture'
        def layer():
            value=Layer();value._mach_native_mixed_fp16=value._mach_native_legacy=True
            value.tp_size=2;value.num_k_heads=16;value.num_v_heads=32;value.head_k_dim=value.head_v_dim=128
            return value
        fake_torch=types.SimpleNamespace(float16='fp16')
        package=types.ModuleType('vllm.model_executor.layers.mamba.gdn');package.qwen_gdn_linear_attn=qwen
        modules={'torch':fake_torch,'vllm.model_executor.layers.mamba.gdn':package}
        node=ast.parse('def _forward_core(self): return _qkv_helper(self)').body[0]
        class Helper:
            def __init__(self,*args):pass
            def direct(self,*args):return None
        with patch.dict('sys.modules',modules),patch.object(gdn,'mixed_fp16',native,create=True),patch.object(views,'_STATE',None), \
             patch.object(views,'digest',side_effect=lambda path:views.QWEN_SHA if path==qwen.__file__ else views.MIXED_SHA), \
             patch.object(views,'source_node',return_value=node),patch.object(views,'transform',return_value=(node,{})),patch.object(views,'Helpers',Helper):
            first=layer();self.assertTrue(views.prepare(first));bound=first._forward_core
            self.assertFalse(views.prepare(first));self.assertIs(first._forward_core,bound)
            second=layer();self.assertTrue(views.prepare(second));self.assertIs(first._forward_core.__func__,second._forward_core.__func__)
            custom=layer();custom._forward_core=object();saved=custom._forward_core;self.assertFalse(views.prepare(custom));self.assertIs(custom._forward_core,saved)
            first._forward_core=object()
            with self.assertRaises(RuntimeError):views.prepare(first)
            original.__code__=(lambda self:42).__code__
            with self.assertRaises(RuntimeError):views.prepare(layer())

    def test_legacy_installs_each_layer_before_device_warmed_return(self):
        node=next(n for n in ast.parse(Path(legacy.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='prepare')
        statements=[ast.unparse(n)for n in node.body]
        self.assertLess(statements.index('direct_norm.prepare(layer)'),statements.index('mixed_qkv_views.prepare(layer)'))
        warmed=next(i for i,n in enumerate(node.body)if isinstance(n,ast.If)and '_WARMED'in ast.unparse(n.test))
        self.assertLess(statements.index('mixed_qkv_views.prepare(layer)'),warmed)
        self.assertEqual(statements[statements.index('direct_norm.prepare(layer)')+1:statements.index('mixed_qkv_views.prepare(layer)')+1],['from . import mixed_qkv_views','mixed_qkv_views.prepare(layer)'])


BODY_SHA = {'compile_function': '5681326806088439bd7a2b74e65cb06bf261702e83149b6fb77df3a280fb28e2', 'private_parts': '740b9b1f4c5aead294315ad036efca8a8956d2f22b65b86d96fff0ab1353ffa5', 'can_view': '5e25dbd6c67786b8a0c84eb94234410eb452123033b462976abeac4ffa9c492a', 'values': '992d24add25b17a4c671a3d94591d4f93d7c2de2454da3b91ebfb917619dc1d7', 'admitted_views': '99567f21f7ca21369254f6631486d2f4ff8ffdbcc87328737f78ae90cbfaf082', 'direct': 'c13b99c33997d241c5465dfd414b1a8b8cfb016de56053744b16ddb55a88f860'}
