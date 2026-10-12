# SPDX-License-Identifier: Apache-2.0
"""Fuse champion SwiGLU and MXFP8 quantization with unchanged BF16 rounding.

Reuse the development branch's producer fusion while preserving the champion's
single final BF16 rounding and FlashInfer's E8M0 conversion. M16/24 producers
participate in PDL. Large prefill calls retain their compiled operations.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

_INSTALLED = False
AUDIT = []
_COUNTS = Counter()


def quantize(x):
    import torch
    import triton
    if (x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda
            or not x.is_contiguous() or x.shape[1] not in (12288, 18432)):
        raise RuntimeError("SwiGLU fusion requires a contiguous supported BF16 gate/up tensor")
    m, width = x.shape
    _COUNTS[("capture" if torch.cuda.is_current_stream_capturing() else "eager", m)] += 1
    if m == 0 or m > 128:
        return None
    k = width // 2
    y = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((triton.cdiv(m, 128)*128*triton.cdiv(k//32, 4)*4,), device=x.device, dtype=torch.uint8)
    groups = 8 if m <= 4 else 32
    _kernel[(m, triton.cdiv(k//32, groups))](x,y,scales,k,groups,m in (16,24),num_warps=4,launch_pdl=m in (16,24))
    return y, scales


import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice, gdc_wait, gdc_launch_dependents


@triton.jit
def _kernel(X,Y,S,K:tl.constexpr,GROUPS:tl.constexpr,PDL:tl.constexpr):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    row=tl.program_id(0)
    groups=tl.program_id(1)*GROUPS+tl.arange(0,GROUPS)
    cols=groups[:,None]*32+tl.arange(0,32)[None,:]
    gate=tl.load(X+row*2*K+cols,cols<K,0).to(tl.float32)
    up=tl.load(X+row*2*K+K+cols,cols<K,0).to(tl.float32)
    # Match the champion compiled producer: no intermediate BF16 SiLU rounding.
    values=((gate/(1.0+libdevice.exp(-gate)))*up).to(tl.bfloat16).to(tl.float32)
    amax=tl.max(tl.abs(values),1)
    rcp_max=tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[tl.full((),448.,tl.float32)], dtype=tl.float32, is_pure=True, pack=1)
    raw=amax*rcp_max
    packed=tl.inline_asm_elementwise("{cvt.rp.satfinite.ue8m0x2.f32 $0, 0f00000000, $1;}", constraints="=h,f", args=[raw], dtype=tl.uint16, is_pure=True, pack=1)
    scale=(packed.to(tl.uint32)&255)
    descale=(scale<<23).to(tl.float32,bitcast=True)
    reciprocal=tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[descale], dtype=tl.float32, is_pure=True, pack=1)
    inverse=tl.where(scale>0,reciprocal,0.)
    quant=tl.minimum(tl.maximum(values*inverse[:,None],-448.),448.).to(tl.float8e4nv)
    tl.store(Y+row*K+cols,quant,cols<K)
    tiles:tl.constexpr=(K//32+3)//4
    offset=(((row//128*tiles+groups//4)*32+row%32)*4+row%128//32)*4+groups%4
    tl.store(S+offset,scale.to(tl.uint8),groups<K//32)


def _producer_hashes(tree):
    found = {}
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Call) or len(node.args) < 2
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
                or not node.args[0].value.startswith("triton_poi_fused_mul_silu_slice_")
                or not isinstance(node.args[1], ast.Constant)
                or not isinstance(node.args[1].value, str)):
            continue
        fn = next(n for n in ast.parse(node.args[1].value).body if isinstance(n, ast.FunctionDef))
        fn.name = "_producer"
        fn.decorator_list = []
        found[node.args[0].value] = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
    return found


def transform(source):
    tree=ast.parse(source)
    runners=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runner']
    if len(runners)!=1:return source,0
    calls=[n for n in runners[0].body if isinstance(n,ast.FunctionDef) and n.name=='call']
    if len(calls)!=1:return source,0
    hashes = _producer_hashes(tree)
    policy = json.loads(Path(__file__).with_name("data").joinpath("swiglu_quant_sources.json").read_text())
    if policy["version"] != 1:
        raise RuntimeError("Unsupported SwiGLU source policy")
    fn=calls[0];lines=source.splitlines(keepends=True);edits={};count=0
    nodes=list(ast.walk(fn))
    for node in nodes:
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
        call=node.value
        if not isinstance(call.func,ast.Attribute) or call.func.attr!='run':continue
        if not isinstance(call.func.value,ast.Name) or not call.func.value.id.startswith('triton_poi_fused_mul_silu_slice_'):continue
        if len(call.args)!=3 or not all(isinstance(a,ast.Name) for a in call.args):continue
        src,dst,_=[a.id for a in call.args]
        hits=[n for n in nodes if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
              and ast.unparse(n.value.func)=='torch.ops.vllm.mxfp8_quantize.default'
              and len(n.value.args)==2 and not n.value.keywords
              and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value is True
              and isinstance(n.value.args[0],ast.Name) and n.value.args[0].id==dst and n.lineno>node.lineno]
        if len(hits)!=1:continue
        if hashes.get(call.func.value.id) not in policy["function_ast_sha256"]:
            raise RuntimeError("Unqualified SwiGLU producer arithmetic")
        q=hits[0]
        if node.lineno!=node.end_lineno or q.lineno!=q.end_lineno:raise RuntimeError('multiline fusion target')
        # Only fold a producer whose output has one actual consumer (quantization).
        refs=[n for n in nodes if isinstance(n,ast.Name) and n.id==dst and isinstance(n.ctx,ast.Load) and n.lineno>node.end_lineno]
        if any(n.lineno!=q.lineno for n in refs):raise RuntimeError('SwiGLU output has extra consumers')
        indent=' '*node.col_offset;tmp=f'_mach_silu_pair_{count}'
        edits[node.lineno-1]=f'{indent}{tmp} = _mach_silu_quant({src})\n{indent}if {tmp} is None:\n    '+lines[node.lineno-1]
        indent=' '*q.col_offset
        edits[q.lineno-1]=f'{indent}{ast.unparse(q.targets[0])} = {tmp} if {tmp} is not None else {ast.unparse(q.value)}\n{indent}del {tmp}\n'
        count+=1
    if not count:return source,0
    for i,text in edits.items():lines[i]=text
    lines.insert(runners[0].lineno-1,'from vllm_mach.mxfp8.swiglu_quant import quantize as _mach_silu_quant\n')
    changed=''.join(lines);ast.parse(changed)
    return changed,count


def install():
    global _INSTALLED
    if _INSTALLED:return
    from torch._inductor.codecache import PyCodeCache
    old=PyCodeCache.load_by_key_path.__func__
    def load(cls,key,path,linemap=None,attrs=None,*,set_sys_modules=None):
        source=Path(path).read_text()
        if 'class Runner' in source and 'triton_poi_fused_mul_silu_slice_' in source:
            changed,count=transform(source)
            if count:
                digest=hashlib.sha256(changed.encode()).hexdigest()
                target=Path(path).with_name('fused_'+digest+'.py')
                if target.exists() and target.read_text() != changed:
                    raise RuntimeError('SwiGLU fusion source hash collision')
                target.write_text(changed)
                AUDIT.append({'source':str(path),'sha256':digest,'pairs':count})
                return old(cls,digest,str(target),None,attrs,set_sys_modules=set_sys_modules)
        return old(cls,key,path,linemap,attrs,set_sys_modules=set_sys_modules)
    PyCodeCache.load_by_key_path=classmethod(load)
    _INSTALLED=True


def receipt(required_rows=()):
    expected = {int(m) for m in required_rows if 0 < m <= 128}
    observed = {m for (phase, m), count in _COUNTS.items() if phase == "capture" and count}
    if required_rows and (not _INSTALLED or not AUDIT or not expected <= observed):
        raise RuntimeError(f"Incomplete SwiGLU/quant capture: expected {expected}, observed {observed}")
    return {"installed": _INSTALLED, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "max_rows": 128, "pdl_rows": [16, 24], "aot_modules": list(AUDIT),
            "counts": [{"phase": phase, "rows": m, "calls": count}
                       for (phase, m), count in sorted(_COUNTS.items())],
            "counter_scope": "Python construction/capture calls, not graph replay counts"}
