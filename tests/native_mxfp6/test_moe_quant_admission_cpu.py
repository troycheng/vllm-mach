"""Native W6A8 admission with per-group scales; no GPU or optional imports."""
import ast
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=ROOT/'tests/fixtures/quark_mxfp6_moe_admission.py'


def make_class(available):
    tree=ast.parse(FIXTURE.read_text())
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id.startswith('k') and n.id[1:2].isupper()}
    class Base:
        def __init__(self,moe,weight,activation):
            if activation not in self.supported_activation_quant_keys:
                raise ValueError('Unsupported activation quant key')
            if weight not in self.supported_weight_quant_keys:
                raise ValueError('Unsupported weight quant key')
            self.moe=moe;self.has_bias=moe.has_bias
            self.activation_quant_key=activation;self.weight_quant_key=weight
    scope={k:k for k in names}
    scope.update(QuarkMoEMethod=Base,is_mxfp6_sm120_moe_available=Mock(return_value=available),
        _WEIGHT_QUANT_KEY_MAP={'mxfp4':'kMxfp4Static','mxfp6_e3m2':'kMxfp6E3M2Static','mxfp6_e2m3':'kMxfp6E2M3Static'},
        _ACTIVATION_QUANT_KEY_MAP={'mxfp8_e4m3':'kMxfp8Dynamic'},
        OCP_MX_Scheme=NS(from_quant_dtype=lambda a,w:'w_'+w+('_a_'+a if a else '')),
        Mxfp4MoeBackend=NS(NONE=object()),
        get_current_vllm_config=lambda:NS(model_config=NS(hf_config=NS(model_type='qwen3_5_moe'))),logger=NS(info_once=Mock()))
    exec(compile(tree,str(FIXTURE),'exec'),scope)
    return scope['QuarkOCP_MX_MoEMethod'],scope


def test_native_per_group_key_retained_without_global_admission_mutation():
    cls,scope=make_class(True)
    native=ModuleType('vllm_mach.mxfp6.moe');native.Mxfp6Sm120Experts=object()
    stock=list(cls.supported_activation_quant_keys)
    with patch.dict('sys.modules',{'vllm_mach.mxfp6.moe':native}):
        instance=cls(NS(has_bias=False),'kMxfp6E3M2Static','kMxfp8Dynamic')
        assert instance.use_mxfp6_sm120_moe and instance.experts_cls is native.Mxfp6Sm120Experts
        assert instance.ocp_mx_scheme=='w_mxfp6_e3m2_a_fp8'
        assert instance.activation_quant_key=='kMxfp8Dynamic'  # remains per-group, not per-tensor
        assert cls.supported_activation_quant_keys==stock and 'kMxfp8Dynamic' not in stock
        scope['is_mxfp6_sm120_moe_available'].return_value=False
        with pytest.raises(ValueError,match='Unsupported activation'):
            cls(NS(has_bias=False),'kMxfp6E3M2Static','kMxfp8Dynamic')


@pytest.mark.parametrize('weight,activation,available',[
    ('kMxfp6E3M2Static','kMxfp8Dynamic',False),
    ('kMxfp4Static','kMxfp8Dynamic',True),
    ('kMxfp6E2M3Static','kMxfp8Dynamic',True),
    ('kMxfp6E3M2Static','kMxfp8Static',True),
])
def test_other_mxfp8_combinations_still_rejected(weight,activation,available):
    cls,scope=make_class(available)
    with pytest.raises(ValueError,match='Unsupported activation'):
        cls(NS(has_bias=False),weight,activation)
    if weight!='kMxfp6E3M2Static' or activation!='kMxfp8Dynamic':
        scope['is_mxfp6_sm120_moe_available'].assert_not_called()


def test_existing_tensor_fp8_native_admission_is_unchanged():
    cls,_=make_class(True)
    native=ModuleType('vllm_mach.mxfp6.moe');native.Mxfp6Sm120Experts=object()
    with patch.dict('sys.modules',{'vllm_mach.mxfp6.moe':native}):
        instance=cls(NS(has_bias=False),'kMxfp6E3M2Static','kFp8DynamicTensorSym')
        assert instance.use_mxfp6_sm120_moe
        assert instance.activation_quant_key=='kFp8DynamicTensorSym'


def test_installed_native_constructor_matches_audited_fixture():
    from vllm.model_executor.layers.quantization.quark.quark_moe import QuarkOCP_MX_MoEMethod
    import textwrap
    actual=ast.parse(textwrap.dedent(inspect.getsource(QuarkOCP_MX_MoEMethod.__init__))).body[0]
    cls=next(n for n in ast.parse(FIXTURE.read_text()).body if isinstance(n,ast.ClassDef))
    expected=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert ast.dump(actual,include_attributes=False)==ast.dump(expected,include_attributes=False)
