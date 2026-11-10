# SPDX-License-Identifier: Apache-2.0
"""Exact division for the admitted BF16/group128 block-FP8 boundary."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _quant(X,Q,S,M:tl.constexpr,K:tl.constexpr,EPS:tl.constexpr,GROUPS:tl.constexpr=4):
    groups=tl.program_id(0)*GROUPS+tl.arange(0,GROUPS)
    cols=tl.arange(0,128)
    n_groups=M*(K//128)
    x=tl.load(X+groups[:,None]*128+cols[None,:],groups[:,None]<n_groups,0).to(tl.float32)
    maximum=tl.maximum(tl.max(tl.abs(x),1),EPS)
    scale=tl.div_rn(maximum,448.0)
    q=tl.minimum(tl.maximum(tl.div_rn(x,scale[:,None]),-448.0),448.0)
    tl.store(Q+groups[:,None]*128+cols[None,:],q,groups[:,None]<n_groups)
    row=groups//(K//128);column=groups%(K//128)
    tl.store(S+column*M+row,scale,groups<n_groups)

def quantize_out(x,q,s):
    assert x.ndim==2 and x.dtype==torch.bfloat16 and x.is_cuda and x.is_contiguous()
    assert q.shape==x.shape and q.dtype==torch.float8_e4m3fn and q.device==x.device and q.is_contiguous()
    assert x.shape[1]%128==0 and s.shape==(x.shape[0],x.shape[1]//128)
    assert s.dtype==torch.float32 and s.device==x.device and s.stride()==(1,x.shape[0])
    _quant[(tr.cdiv(x.numel()//128,4),)](x,q,s,x.shape[0],x.shape[1],1e-10,num_warps=4,enable_fp_fusion=False)
