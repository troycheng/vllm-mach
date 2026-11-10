"""Finite BF16 GDN arithmetic and preserved FP32/stock dispatch contracts."""
import ast, copy, hashlib, json, os
from pathlib import Path
import pytest
from vllm_mach import gdn_prep
from vllm_mach.mxfp8 import ordered_reduction as ordered
from test_mxfp8_ordered_reduction_cpu import compiler, templates, decode
ROOT=Path(__file__).resolve().parents[1]


def scope(monkeypatch, dtype='bfloat16', ordered_flag='1', mode='production'):
    monkeypatch.setenv('VLLM_MACH_PROFILE','qwen35-4b-mxfp8-champion-v1')
    monkeypatch.setenv('VLLM_MACH_MXFP8_SSM_DTYPE',dtype)
    monkeypatch.setenv('VLLM_MACH_MXFP8_ORDERED_BF16_SSM',ordered_flag)
    monkeypatch.setenv('VLLM_MACH_MXFP8_MODE','production' if mode=='production' else 'quality')
    if mode=='production':monkeypatch.delenv('VLLM_MACH_MXFP8_QUALITY_ROWS',raising=False)
    else:monkeypatch.setenv('VLLM_MACH_MXFP8_QUALITY_ROWS',mode[-2:])


@pytest.mark.parametrize('mode',['production','quality32','quality64'])
def test_bf16_generated_norm_exact_source_and_old_component_tree(monkeypatch,mode):
    scope(monkeypatch,mode=mode);m=compiler();p=m.load_policy()
    row=next(r for r in p['bindings'] if r.get('ordered_reduction',{}).get('version')==2)
    source,fn=templates(m)[row['boundary']['function_sha256']]
    tree=ast.parse(source)
    for n in ast.walk(tree):
        if isinstance(n,ast.keyword)and n.arg=='size_hints':n.value=ast.parse(repr(row['boundary']['size_hints']),mode='eval').body
    source=ast.unparse(ast.fix_missing_locations(tree))
    transformed=m._contextual_triton(lambda self,name,s:s)(None,fn.name,source)
    assert m.canonical_function_sha256(transformed,fn.name)==row['ordered_reduction']['function_sha256']
    target=next(n for n in ast.parse(transformed).body if isinstance(n,ast.FunctionDef)and n.name==fn.name)
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef))
    branch=next(n for n in target.body if isinstance(n,ast.If)and ast.unparse(n.test)=='XBLOCK == 32')
    reverted=copy.deepcopy(target);reverted.body[reverted.body.index(next(n for n in reverted.body if isinstance(n,ast.If)and ast.unparse(n.test)=='XBLOCK == 32'))]=copy.deepcopy(branch.orelse[0])
    assert m.canonical_function_sha256(ast.unparse(reverted))==row['boundary']['function_sha256']
    assert ast.dump(original.body[-1])==ast.dump(target.body[-1])
    b=decode(row['boundary']);meta=dict(signature=b['signature'],constants=b['constants'],configs=b['specialization'],device=b['architecture'],**b['compiler_flags'])
    assert m.select_recipe(ast.unparse(target),b['size_hints'],meta,kernel_alias=row['kernel_alias'],kernel_site=row['kernel_site'],require_ordered=True)['id']==row['id']
    with pytest.raises(m.CompileChoiceMismatch,match='bypassed'):
        m.select_recipe(ast.unparse(original),b['size_hints'],meta,kernel_alias=row['kernel_alias'],kernel_site=row['kernel_site'],require_ordered=True)
    template=next(r for r in p['bindings']if r['construction_kind']=='template'and r['boundary']==row['boundary'])
    assert m.select_recipe(ast.unparse(original),b['size_hints'],meta,kernel_alias=template['kernel_alias'],kernel_site=template['kernel_site'],require_ordered=True)['id']==template['id']
    assert hashlib.sha256((ROOT/'src/vllm_mach/mxfp8/_ordered_reduction_gdn_kernel.py').read_bytes()).hexdigest()=='4c4c03b704d04c8196fba7d1f14c07f9b538302527f9d2fbdc08a595bd56396a'


@pytest.mark.parametrize('dtype,flag,enabled',[('bfloat16','1',True),('bfloat16','0',False),('float32','1',False)])
@pytest.mark.parametrize('mode',['production','quality32','quality64'])
def test_fp32_and_stock_gdn_keep_original_source_and_prep(monkeypatch,dtype,flag,enabled,mode):
    scope(monkeypatch,dtype,flag,mode);m=compiler();p=m.load_policy()
    row=next(r for r in p['bindings']if r.get('ordered_reduction',{}).get('version')==2)
    assert (m._ordered_spec(row)is not None)==enabled
    assert gdn_prep.selected('qwen35-4b-mxfp8-champion-v1',heads=16,value_heads=32,head_dim=128,value_dim=128,bf16=True,apply_l2norm=True,output_g_exp=False)==enabled
    if not enabled:
        source,fn=templates(m)[row['boundary']['function_sha256']];tree=ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.keyword)and n.arg=='size_hints':n.value=ast.parse(repr(row['boundary']['size_hints']),mode='eval').body
        source=ast.unparse(ast.fix_missing_locations(tree));assert m._contextual_triton(lambda self,name,s:s)(None,fn.name,source)==source


def test_selected_mxfp8_4b_actual_prep_binding():
    import subprocess, sys
    for dtype,flag,enabled in [('float32','1',False),('bfloat16','0',False),('bfloat16','1',True)]:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','VLLM_PLUGINS':'',
             'VLLM_MACH_PROFILE':'qwen35-4b-mxfp8-champion-v1',
             'VLLM_MACH_MXFP8_SSM_DTYPE':dtype,'VLLM_MACH_MXFP8_ORDERED_BF16_SSM':flag,
             'PYTHONPATH':str(ROOT/'src')}
        subprocess.run([sys.executable,'-c',f'''
import torch
from vllm_mach import gdn_prep
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep as stock
gdn_prep.verify_binding()
assert gdn.fused_post_conv_prep is (gdn_prep.fused_post_conv_prep if {enabled!r} else stock)
assert not torch.cuda.is_initialized()
'''],env=env,check=True,timeout=90)
