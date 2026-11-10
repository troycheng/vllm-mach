"""R128/X32/w2 local FP32/FMA order from SM120 PTX."""
import triton
import triton.language as tl
@triton.jit
def add(a,b):return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)
@triton.jit
def mul(a,b):return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)
@triton.jit
def fma(a,b,c):return tl.inline_asm_elementwise('fma.rn.f32 $0, $1, $2, $3;',constraints='=f,f,f,f',args=[a,b,c],dtype=tl.float32,is_pure=True,pack=1)
@triton.jit
def old_norm128_sum(x,XBLOCK:tl.constexpr,R0_BLOCK:tl.constexpr):
 tl.static_assert(XBLOCK==32 and R0_BLOCK==128)
 q0246,q1357=tl.split(tl.reshape(x,(XBLOCK,R0_BLOCK//8,2,2,2)))
 q04,q26=tl.split(q0246);q15,q37=tl.split(q1357)
 q0,q4=tl.split(q04);q2,q6=tl.split(q26);q1,q5=tl.split(q15);q3,q7=tl.split(q37)
 local=fma(q0,q0,mul(q1,q1))
 local=fma(q2,q2,local)
 local=add(mul(q3,q3),local)
 local=fma(q4,q4,local)
 local=add(mul(q5,q5),local)
 local=fma(q6,q6,local)
 local=add(mul(q7,q7),local)
 return tl.sum(local,1)[:,None]
