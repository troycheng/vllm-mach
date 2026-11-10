# SPDX-License-Identifier: Apache-2.0
"""SM120 local4 FP32 reductions with the qualified Triton3.7 rounding order.

Callers must bind the actual local4 layout; BF16 local8 state reductions must
use a separately qualified helper. These functions do not select a profile.
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
