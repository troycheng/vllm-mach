# SPDX-License-Identifier: Apache-2.0
"""SM120 K128 local4 square/local8 dot reductions in the Triton 3.7 order."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _gl_rn_mul(x,y):
 return gl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,y],dtype=gl.float32,is_pure=True,pack=1)

@gluon.jit
def _gl_rn_fma(x,y,z):
 return gl.inline_asm_elementwise('fma.rn.f32 $0, $1, $2, $3;',constraints='=f,f,f,f',args=[x,y,z],dtype=gl.float32,is_pure=True,pack=1)

@gluon.jit
def _gl_old_square_128(x):
 x02,x13=gl.split(gl.reshape(x,(32,2,2)))
 x0,x2=gl.split(x02);x1,x3=gl.split(x13)
 local=_gl_rn_fma(x0,x0,_gl_rn_mul(x1,x1))
 local=_gl_rn_fma(x2,x2,local);local=_gl_rn_fma(x3,x3,local)
 return gl.sum(local,0)

@gluon.jit
def _gl_old_dot_128_8local(x,y,BV:gl.constexpr):
 gl.static_assert(BV==32)
 x,y=gl.broadcast(x,y)
 x0246,x1357=gl.split(gl.reshape(x,(BV,16,2,2,2)))
 x04,x26=gl.split(x0246);x15,x37=gl.split(x1357)
 x0,x4=gl.split(x04);x2,x6=gl.split(x26)
 x1,x5=gl.split(x15);x3,x7=gl.split(x37)
 y0246,y1357=gl.split(gl.reshape(y,(BV,16,2,2,2)))
 y04,y26=gl.split(y0246);y15,y37=gl.split(y1357)
 y0,y4=gl.split(y04);y2,y6=gl.split(y26)
 y1,y5=gl.split(y15);y3,y7=gl.split(y37)
 local=_gl_rn_fma(x0,y0,_gl_rn_mul(x1,y1))
 local=_gl_rn_fma(x2,y2,local);local=_gl_rn_fma(x3,y3,local)
 local=_gl_rn_fma(x4,y4,local);local=_gl_rn_fma(x5,y5,local)
 local=_gl_rn_fma(x6,y6,local);local=_gl_rn_fma(x7,y7,local)
 return gl.convert_layout(gl.sum(local,1),gl.SliceLayout(1,x.type.layout))
