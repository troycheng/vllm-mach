"""Finite ordered norm source/selector/helper guards; no Torch/Triton needed."""
import ast,copy,importlib.util,json,os,types
from pathlib import Path
import pytest
from vllm_mach.mxfp8 import ordered_reduction as ordered
ROOT=Path(__file__).resolve().parents[1]


def compiler():
    s=importlib.util.spec_from_file_location('ordered_norm_contract',ROOT/'src/vllm_mach/mxfp8/compile_choices.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def templates(m):
    result={}
    for source in json.loads((ROOT/'tests/data/inductor033-ordered-norms.json').read_text()):
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef))
        result[m.canonical_function_sha256(source,fn.name)]=(source,fn)
    return result


def decode(value):
    if isinstance(value,list):return [decode(v) for v in value]
    if isinstance(value,dict):
        if set(value)=={'__tuple__'}:return tuple(decode(v) for v in value['__tuple__'])
        if set(value)=={'__mapping__'}:return {decode(k):decode(v) for k,v in value['__mapping__']}
        return {k:decode(v) for k,v in value.items()}
    return value


@pytest.mark.parametrize('mode,count',[('production',4),('quality32',4),('quality64',3)])
def test_all_policies_keep_exact_original_and_transformed_contract(monkeypatch,mode,count):
    monkeypatch.setenv('VLLM_MACH_MXFP8_MODE','production' if mode=='production' else 'quality')
    if mode=='production':monkeypatch.delenv('VLLM_MACH_MXFP8_QUALITY_ROWS',raising=False)
    else:monkeypatch.setenv('VLLM_MACH_MXFP8_QUALITY_ROWS',mode[-2:])
    m=compiler();policy=m.load_policy();sources=templates(m);seen=[]
    for row in policy['bindings']:
        if 'ordered_reduction' not in row:continue
        spec=row['ordered_reduction'];source,fn=sources[row['boundary']['function_sha256']]
        changed=ordered.transform(source,fn.name,spec);target=next(n for n in ast.parse(changed).body if isinstance(n,ast.FunctionDef) and n.name==fn.name)
        assert ast.dump(target.body[-1],include_attributes=False)==ast.dump(fn.body[-1],include_attributes=False)
        assert m.canonical_function_sha256(changed,fn.name)==spec['function_sha256']
        b=decode(row['boundary']);meta=dict(signature=b['signature'],constants=b['constants'],configs=b['specialization'],device=b['architecture'],**b['compiler_flags'])
        kwargs=dict(kernel_alias=row['kernel_alias'],kernel_site=row['kernel_site'],kernel_context={row['context_semantic_key']} if 'context_semantic_key' in row else None,require_ordered=True)
        selected=m.select_recipe(ast.unparse(target),b['size_hints'],meta,**kwargs)
        assert selected['id']==row['id'] and selected['selected_fields']==row['selected_fields']
        with pytest.raises(m.CompileChoiceMismatch,match='bypassed'):
            m.select_recipe(source,b['size_hints'],meta,**kwargs)
        with pytest.raises(m.CompileChoiceMismatch):
            m.select_recipe(ast.unparse(target),b['size_hints'],{**meta,'enable_fp_fusion':False},**kwargs)
        if 'context_semantic_key' in row:
            with pytest.raises(m.CompileChoiceMismatch):
                m.select_recipe(ast.unparse(target),b['size_hints'],meta,**{**kwargs,'kernel_context':{'wrong'}})
        seen.append(row['id'])
    assert len(seen)==count


def test_helper_source_and_actual_jit_dependency_are_guarded():
    m=compiler();policy=json.loads((ROOT/'src/vllm_mach/mxfp8/data/compile_choices.json').read_text());row=next(r for r in policy['bindings'] if 'ordered_reduction' in r)
    spec=copy.deepcopy(row['ordered_reduction']);spec['helper_sha256']='0'*64
    with pytest.raises(ValueError,match='helper source'):ordered.validate(spec,row['boundary'],row['selected_fields'])
    values={node.name:types.SimpleNamespace(src=ast.unparse(node)) for node in ast.parse(ordered.helper_source()).body if isinstance(node,ast.FunctionDef)}
    fake=types.SimpleNamespace(fn=types.SimpleNamespace(__globals__=values))
    ordered.validate_globals(fake,m.canonical_function_sha256)
    values['_old_ordered_add'].src=values['_old_ordered_add'].src.replace('add.rn.f32','mul.rn.f32')
    with pytest.raises(RuntimeError,match='JIT helper differs'):ordered.validate_globals(fake,m.canonical_function_sha256)


def test_unknown_tile_or_ordered_spec_is_rejected():
    policy=json.loads((ROOT/'src/vllm_mach/mxfp8/data/compile_choices_quality32.json').read_text());row=next(r for r in policy['bindings'] if 'ordered_reduction' in r)
    for fields in [{**row['selected_fields'],'R0_BLOCK':2048},{**row['selected_fields'],'num_warps':4}]:
        with pytest.raises(ValueError,match='tile differs'):ordered.validate(row['ordered_reduction'],row['boundary'],fields)
    with pytest.raises(ValueError,match='Unknown'):
        ordered.validate({**row['ordered_reduction'],'unknown':True},row['boundary'],row['selected_fields'])


def test_gpu_proven_quality32_function_variants_keep_their_source_identity():
    policy=json.loads((ROOT/'src/vllm_mach/mxfp8/data/compile_choices_quality32.json').read_text())
    proofs=json.loads((ROOT/'tests/data/inductor033-ordered-norm-proofs.json').read_text())
    for row in policy['bindings']:
        spec=row.get('ordered_reduction')
        if spec is None:continue
        proof=next(p for p in proofs if p['original_function_sha256']==row['boundary']['function_sha256'] and p['group']==spec['group'])
        assert spec['chunk_accumulation']==proof['chunk_accumulation']
        assert spec['function_sha256']==proof['transformed_function_sha256']


def test_production_three_input_chunk_retains_separate_multiply_and_add():
    m=compiler();policy=json.loads((ROOT/'src/vllm_mach/mxfp8/data/compile_choices.json').read_text())
    row=next(r for r in policy['bindings'] if r['id']=='generated_923de6c9cac2')
    assert row['ordered_reduction']['chunk_accumulation']=='separate'
    source,fn=templates(m)[row['boundary']['function_sha256']]
    changed=ordered.transform(source,fn.name,row['ordered_reduction'])
    target=next(n for n in ast.parse(changed).body if isinstance(n,ast.FunctionDef) and n.name==fn.name)
    calls=[ast.unparse(n.func) for n in ast.walk(target) if isinstance(n,ast.Call)]
    assert '_old_ordered_mul' in calls and '_old_ordered_add' in calls and 'tl.fma' not in calls
