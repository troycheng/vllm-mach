# SPDX-License-Identifier: Apache-2.0
"""Frozen M16x2048 BF16 global-scale reduction."""
import triton
import triton.language as tl

@triton.jit
def scalar_finalize(max_abs,any_nan):
    minimum=tl.full((),1.0e-12,tl.float32)
    one=tl.full((),1.0,tl.float32)
    scale=tl.full((),2688.0,tl.float32)
    clamped=tl.maximum(max_abs,minimum)
    reciprocal=tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[one,clamped],dtype=tl.float32,is_pure=True,pack=1)
    scaled=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[reciprocal,scale],dtype=tl.float32,is_pure=True,pack=1)
    return tl.where(any_nan | (max_abs<=0),one,scaled)

@triton.jit
def reduce_scale(X,Y):
    index=tl.arange(0,32768)
    value=tl.load(X+index).to(tl.float32)
    is_nan=value!=value
    absolute=tl.abs(value)
    maximum=tl.max(tl.where(is_nan,0.0,absolute),axis=0)
    any_nan=tl.max(is_nan.to(tl.int32),axis=0)!=0
    tl.store(Y,scalar_finalize(maximum,any_nan))
