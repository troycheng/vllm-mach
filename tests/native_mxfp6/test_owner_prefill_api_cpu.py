"""CPU source-extracted owner custom-op ABI contract; no GPU/runtime import proof."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[2] / 'src/vllm_mach/mxfp6/sm120_owner_prefill.py'


def fixture(metadata):
    tree=ast.parse(SOURCE.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gdn_forward')
    calls=[]
    scope={'torch':torch,'F':F,'apply_mxfp8_weight':Mock(return_value=torch.zeros((512,5),dtype=torch.bfloat16)),
           'packed_weight':lambda _:object(), 'gather_bytes':lambda *a:torch.zeros((2,512*48*2),dtype=torch.uint8)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),scope)
    module=NS(prefix='layer1',in_proj_qkvz=object(),num_v_heads=2,tp_size=2,head_v_dim=2,
              out_proj=lambda x:(x.clone(),None))
    state=NS(m=512,rank=0,own=slice(0,256),other=slice(256,512),verify=False,
             ba_weights={id(module):(torch.zeros((24,4),dtype=torch.bfloat16),)*2})
    normalized=NS(norm=torch.ones((512,4),dtype=torch.bfloat16),prepared=object())
    forward=ModuleType('vllm.forward_context');forward.get_forward_context=lambda:NS(attn_metadata=metadata)
    gdn=ModuleType('vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn');gdn._encode_layer_name=lambda x:x
    def op(qkvz,ba,core,hidden_states,layer_name):
        calls.append((qkvz,ba,core,hidden_states,layer_name));core.fill_(3)
    patches=patch.dict('sys.modules',{'vllm.forward_context':forward,
        'vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn':gdn})
    return scope,module,state,normalized,calls,patches,op


@pytest.mark.parametrize('metadata',[None,{'layer1':NS(num_prefills=1,num_decodes=0)},
                                     {'layer1':NS(num_prefills=1,num_decodes=4)}])
def test_owner_prefill_passes_new_hidden_tensor_only_when_decode_cannot_use_it(metadata):
    scope,module,state,norm,calls,modules,op=fixture(metadata)
    with modules,patch.object(torch.ops.vllm,'qwen_gdn_attention_core_fused_norm_packed',op,create=True):
        output=scope['gdn_forward'](state,module,norm,'test')
    assert len(calls)==1 and calls[0][3] is norm.norm and calls[0][4]=='layer1'
    assert calls[0][0].shape==(512,5) and calls[0][1].shape==(512,48)
    assert bool((norm.norm[state.own]==1).all()) and bool((norm.norm[state.other]==0).all())
    assert output.shape==(512,2) and bool((output==3).all())


def test_owner_rejects_decode_before_mutating_partial_hidden_or_launching():
    scope,module,state,norm,calls,modules,op=fixture({'layer1':NS(num_prefills=0,num_decodes=512)})
    original=norm.norm.clone()
    with modules,patch.object(torch.ops.vllm,'qwen_gdn_attention_core_fused_norm_packed',op,create=True):
        with pytest.raises(RuntimeError,match='partial hidden rows cannot feed fused decode'):
            scope['gdn_forward'](state,module,norm,'test')
    scope['apply_mxfp8_weight'].assert_not_called()
    assert not calls and torch.equal(norm.norm,original)
