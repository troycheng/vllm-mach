# SPDX-License-Identifier: Apache-2.0
"""SM120 local4 and local8 reductions in the qualified Triton 3.7 order.

Callers bind the qualified layout and choose its local4 or local8 dot helper.
These functions do not select a profile.
"""
from vllm.triton_utils import tl, triton

@triton.jit
def _rn_mul(x,y):
 return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,y],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _rn_fma(x,y,z):
 return tl.inline_asm_elementwise('fma.rn.f32 $0, $1, $2, $3;',constraints='=f,f,f,f',args=[x,y,z],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _old_square_128(x):
 x02,x13=tl.split(tl.reshape(x,(32,2,2)))
 x0,x2=tl.split(x02)
 x1,x3=tl.split(x13)
 local=_rn_fma(x0,x0,_rn_mul(x1,x1))
 local=_rn_fma(x2,x2,local)
 local=_rn_fma(x3,x3,local)
 return tl.sum(local,0)

@triton.jit
def _old_dot_128(x,y,BV:tl.constexpr):
 tl.static_assert(BV==32)
 y=tl.broadcast_to(y,(BV,128))
 x02,x13=tl.split(tl.reshape(x,(BV,32,2,2)))
 x0,x2=tl.split(x02)
 x1,x3=tl.split(x13)
 y02,y13=tl.split(tl.reshape(y,(BV,32,2,2)))
 y0,y2=tl.split(y02)
 y1,y3=tl.split(y13)
 local=_rn_fma(x0,y0,_rn_mul(x1,y1))
 local=_rn_fma(x2,y2,local)
 local=_rn_fma(x3,y3,local)
 return tl.sum(local,1)


@triton.jit
def _old_dot_128_8local(x, y, BV: tl.constexpr):
    tl.static_assert(BV == 32)
    y = tl.broadcast_to(y, (BV, 128))
    x0246, x1357 = tl.split(tl.reshape(x, (BV, 16, 2, 2, 2)))
    x04, x26 = tl.split(x0246)
    x15, x37 = tl.split(x1357)
    x0, x4 = tl.split(x04)
    x2, x6 = tl.split(x26)
    x1, x5 = tl.split(x15)
    x3, x7 = tl.split(x37)
    y0246, y1357 = tl.split(tl.reshape(y, (BV, 16, 2, 2, 2)))
    y04, y26 = tl.split(y0246)
    y15, y37 = tl.split(y1357)
    y0, y4 = tl.split(y04)
    y2, y6 = tl.split(y26)
    y1, y5 = tl.split(y15)
    y3, y7 = tl.split(y37)
    local = _rn_fma(x0, y0, _rn_mul(x1, y1))
    local = _rn_fma(x2, y2, local)
    local = _rn_fma(x3, y3, local)
    local = _rn_fma(x4, y4, local)
    local = _rn_fma(x5, y5, local)
    local = _rn_fma(x6, y6, local)
    local = _rn_fma(x7, y7, local)
    return tl.sum(local, 1)
