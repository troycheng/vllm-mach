# SPDX-License-Identifier: Apache-2.0

"""Reset W4 metadata on newly assigned scheduler slots."""
from vllm.triton_utils import tl, triton

@triton.jit
def _reset_meta(IDS, AGE, PREFIX, P: tl.constexpr,
                INDEX_STRIDE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(IDS + row*INDEX_STRIDE).to(tl.int32)
    offs = tl.arange(0, B)
    valid = (slot > 0) & (slot < P) & (offs < B)
    tl.store(AGE + slot * B + offs, 0, mask=valid)
    tl.store(PREFIX + slot * B + offs, 1.0, mask=valid)

def reset_metadata(indices, scratch, pages):
    if indices.numel():
        _reset_meta[(indices.numel(),)](indices,scratch.age,scratch.prefix,
                                       pages,indices.stride(0),64,num_warps=4)
