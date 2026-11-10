# SPDX-License-Identifier: Apache-2.0
# Finite SM120 2B legacy merge arithmetic; stock guards stay intact.
# New padding/query/seq guards and empty-segment exclusion stay intact.
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import cdiv_fn, find_seq_idx
float8_info = torch.finfo(current_platform.fp8_dtype())

@triton.jit
def reduce_segments_explicit_old_lane(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    # Graph padding must be rejected before sequence lookup or scratch access.
    if query_token_idx >= tl.load(query_start_len_ptr + num_seqs):
        return

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        return

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    # Empty segments must not set the maximum. Unvisited segments retain
    # the M=-inf,L=1 initial state.
    valid_segment = segm_mask & (segm_expsum > 0) & (segm_max != float("-inf"))
    # Keep invalid/empty segments excluded, but put their predicate on the
    # staging loads. This restores the old exp/multiply/reduce expression
    # boundary for valid segments, rather than wrapping exp in tl.where.
    segm_mask = segm_mask & valid_segment
    segm_max = tl.where(segm_mask, segm_max, float("-inf"))
    overall_max = tl.max(segm_max)
    # All-invalid rows must not form exp(-inf - -inf). Positive-length valid
    # rows retain exactly their original finite maximum.
    overall_max = tl.where(overall_max == float("-inf"), 0.0, overall_max)

    # Deliberately preserve the old numeric expression at this boundary.
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    raw_segm_expsum = segm_expsum

    # Explicit old numerator tree with direct 1D output-dimension loads.
    tl.static_assert(NUM_SEGMENTS_PER_SEQ == 16)
    tl.static_assert(HEAD_SIZE == 256 and HEAD_SIZE_PADDED == 256)
    tl.static_assert(not USE_FP8)
    segment_scale = tl.exp(segm_max - overall_max)
    dims = tl.arange(0, HEAD_SIZE_PADDED)
    stage_base = query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
    s0 = tl.sum(tl.gather(segment_scale, tl.full((1,), 0, tl.int32), 0), 0)
    valid0 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 0, tl.int32), 0), 0) > 0
    o0 = tl.load(segm_output_ptr + stage_base + 0 * HEAD_SIZE_PADDED + dims, mask=valid0 & (dims < HEAD_SIZE), other=0.0)
    s1 = tl.sum(tl.gather(segment_scale, tl.full((1,), 1, tl.int32), 0), 0)
    valid1 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 1, tl.int32), 0), 0) > 0
    o1 = tl.load(segm_output_ptr + stage_base + 1 * HEAD_SIZE_PADDED + dims, mask=valid1 & (dims < HEAD_SIZE), other=0.0)
    s2 = tl.sum(tl.gather(segment_scale, tl.full((1,), 2, tl.int32), 0), 0)
    valid2 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 2, tl.int32), 0), 0) > 0
    o2 = tl.load(segm_output_ptr + stage_base + 2 * HEAD_SIZE_PADDED + dims, mask=valid2 & (dims < HEAD_SIZE), other=0.0)
    s3 = tl.sum(tl.gather(segment_scale, tl.full((1,), 3, tl.int32), 0), 0)
    valid3 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 3, tl.int32), 0), 0) > 0
    o3 = tl.load(segm_output_ptr + stage_base + 3 * HEAD_SIZE_PADDED + dims, mask=valid3 & (dims < HEAD_SIZE), other=0.0)
    s4 = tl.sum(tl.gather(segment_scale, tl.full((1,), 4, tl.int32), 0), 0)
    valid4 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 4, tl.int32), 0), 0) > 0
    o4 = tl.load(segm_output_ptr + stage_base + 4 * HEAD_SIZE_PADDED + dims, mask=valid4 & (dims < HEAD_SIZE), other=0.0)
    s5 = tl.sum(tl.gather(segment_scale, tl.full((1,), 5, tl.int32), 0), 0)
    valid5 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 5, tl.int32), 0), 0) > 0
    o5 = tl.load(segm_output_ptr + stage_base + 5 * HEAD_SIZE_PADDED + dims, mask=valid5 & (dims < HEAD_SIZE), other=0.0)
    s6 = tl.sum(tl.gather(segment_scale, tl.full((1,), 6, tl.int32), 0), 0)
    valid6 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 6, tl.int32), 0), 0) > 0
    o6 = tl.load(segm_output_ptr + stage_base + 6 * HEAD_SIZE_PADDED + dims, mask=valid6 & (dims < HEAD_SIZE), other=0.0)
    s7 = tl.sum(tl.gather(segment_scale, tl.full((1,), 7, tl.int32), 0), 0)
    valid7 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 7, tl.int32), 0), 0) > 0
    o7 = tl.load(segm_output_ptr + stage_base + 7 * HEAD_SIZE_PADDED + dims, mask=valid7 & (dims < HEAD_SIZE), other=0.0)
    s8 = tl.sum(tl.gather(segment_scale, tl.full((1,), 8, tl.int32), 0), 0)
    valid8 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 8, tl.int32), 0), 0) > 0
    o8 = tl.load(segm_output_ptr + stage_base + 8 * HEAD_SIZE_PADDED + dims, mask=valid8 & (dims < HEAD_SIZE), other=0.0)
    s9 = tl.sum(tl.gather(segment_scale, tl.full((1,), 9, tl.int32), 0), 0)
    valid9 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 9, tl.int32), 0), 0) > 0
    o9 = tl.load(segm_output_ptr + stage_base + 9 * HEAD_SIZE_PADDED + dims, mask=valid9 & (dims < HEAD_SIZE), other=0.0)
    s10 = tl.sum(tl.gather(segment_scale, tl.full((1,), 10, tl.int32), 0), 0)
    valid10 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 10, tl.int32), 0), 0) > 0
    o10 = tl.load(segm_output_ptr + stage_base + 10 * HEAD_SIZE_PADDED + dims, mask=valid10 & (dims < HEAD_SIZE), other=0.0)
    s11 = tl.sum(tl.gather(segment_scale, tl.full((1,), 11, tl.int32), 0), 0)
    valid11 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 11, tl.int32), 0), 0) > 0
    o11 = tl.load(segm_output_ptr + stage_base + 11 * HEAD_SIZE_PADDED + dims, mask=valid11 & (dims < HEAD_SIZE), other=0.0)
    s12 = tl.sum(tl.gather(segment_scale, tl.full((1,), 12, tl.int32), 0), 0)
    valid12 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 12, tl.int32), 0), 0) > 0
    o12 = tl.load(segm_output_ptr + stage_base + 12 * HEAD_SIZE_PADDED + dims, mask=valid12 & (dims < HEAD_SIZE), other=0.0)
    s13 = tl.sum(tl.gather(segment_scale, tl.full((1,), 13, tl.int32), 0), 0)
    valid13 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 13, tl.int32), 0), 0) > 0
    o13 = tl.load(segm_output_ptr + stage_base + 13 * HEAD_SIZE_PADDED + dims, mask=valid13 & (dims < HEAD_SIZE), other=0.0)
    s14 = tl.sum(tl.gather(segment_scale, tl.full((1,), 14, tl.int32), 0), 0)
    valid14 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 14, tl.int32), 0), 0) > 0
    o14 = tl.load(segm_output_ptr + stage_base + 14 * HEAD_SIZE_PADDED + dims, mask=valid14 & (dims < HEAD_SIZE), other=0.0)
    s15 = tl.sum(tl.gather(segment_scale, tl.full((1,), 15, tl.int32), 0), 0)
    valid15 = tl.sum(tl.gather(segm_mask.to(tl.int32), tl.full((1,), 15, tl.int32), 0), 0) > 0
    o15 = tl.load(segm_output_ptr + stage_base + 15 * HEAD_SIZE_PADDED + dims, mask=valid15 & (dims < HEAD_SIZE), other=0.0)
    # Old scalar TL sum used asymmetric FMA at XOR8 before XOR4/2/1.
    # Preserve both lane groups; old four-dimension divide layout selects
    # group B with output dimension bit5, not the new layout bit4.
    l0 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 0, tl.int32), 0), 0)
    p0 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s0,l0], dtype=tl.float32, is_pure=True, pack=1)
    l1 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 1, tl.int32), 0), 0)
    p1 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s1,l1], dtype=tl.float32, is_pure=True, pack=1)
    l2 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 2, tl.int32), 0), 0)
    p2 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s2,l2], dtype=tl.float32, is_pure=True, pack=1)
    l3 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 3, tl.int32), 0), 0)
    p3 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s3,l3], dtype=tl.float32, is_pure=True, pack=1)
    l4 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 4, tl.int32), 0), 0)
    p4 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s4,l4], dtype=tl.float32, is_pure=True, pack=1)
    l5 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 5, tl.int32), 0), 0)
    p5 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s5,l5], dtype=tl.float32, is_pure=True, pack=1)
    l6 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 6, tl.int32), 0), 0)
    p6 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s6,l6], dtype=tl.float32, is_pure=True, pack=1)
    l7 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 7, tl.int32), 0), 0)
    p7 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s7,l7], dtype=tl.float32, is_pure=True, pack=1)
    l8 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 8, tl.int32), 0), 0)
    p8 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s8,l8], dtype=tl.float32, is_pure=True, pack=1)
    l9 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 9, tl.int32), 0), 0)
    p9 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s9,l9], dtype=tl.float32, is_pure=True, pack=1)
    l10 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 10, tl.int32), 0), 0)
    p10 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s10,l10], dtype=tl.float32, is_pure=True, pack=1)
    l11 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 11, tl.int32), 0), 0)
    p11 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s11,l11], dtype=tl.float32, is_pure=True, pack=1)
    l12 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 12, tl.int32), 0), 0)
    p12 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s12,l12], dtype=tl.float32, is_pure=True, pack=1)
    l13 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 13, tl.int32), 0), 0)
    p13 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s13,l13], dtype=tl.float32, is_pure=True, pack=1)
    l14 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 14, tl.int32), 0), 0)
    p14 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s14,l14], dtype=tl.float32, is_pure=True, pack=1)
    l15 = tl.sum(tl.gather(raw_segm_expsum, tl.full((1,), 15, tl.int32), 0), 0)
    p15 = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s15,l15], dtype=tl.float32, is_pure=True, pack=1)
    a0 = tl.fma(s0, l0, p8)
    a1 = tl.fma(s1, l1, p9)
    a2 = tl.fma(s2, l2, p10)
    a3 = tl.fma(s3, l3, p11)
    a4 = tl.fma(s4, l4, p12)
    a5 = tl.fma(s5, l5, p13)
    a6 = tl.fma(s6, l6, p14)
    a7 = tl.fma(s7, l7, p15)
    ax4_0 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a0,a4], dtype=tl.float32, is_pure=True, pack=1)
    ax4_1 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a1,a5], dtype=tl.float32, is_pure=True, pack=1)
    ax4_2 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a2,a6], dtype=tl.float32, is_pure=True, pack=1)
    ax4_3 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a3,a7], dtype=tl.float32, is_pure=True, pack=1)
    ax2_0 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[ax4_0,ax4_2], dtype=tl.float32, is_pure=True, pack=1)
    ax2_1 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[ax4_1,ax4_3], dtype=tl.float32, is_pure=True, pack=1)
    den_a = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[ax2_0,ax2_1], dtype=tl.float32, is_pure=True, pack=1)
    b0 = tl.fma(s8, l8, p0)
    b1 = tl.fma(s9, l9, p1)
    b2 = tl.fma(s10, l10, p2)
    b3 = tl.fma(s11, l11, p3)
    b4 = tl.fma(s12, l12, p4)
    b5 = tl.fma(s13, l13, p5)
    b6 = tl.fma(s14, l14, p6)
    b7 = tl.fma(s15, l15, p7)
    bx4_0 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[b0,b4], dtype=tl.float32, is_pure=True, pack=1)
    bx4_1 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[b1,b5], dtype=tl.float32, is_pure=True, pack=1)
    bx4_2 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[b2,b6], dtype=tl.float32, is_pure=True, pack=1)
    bx4_3 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[b3,b7], dtype=tl.float32, is_pure=True, pack=1)
    bx2_0 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[bx4_0,bx4_2], dtype=tl.float32, is_pure=True, pack=1)
    bx2_1 = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[bx4_1,bx4_3], dtype=tl.float32, is_pure=True, pack=1)
    den_b = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[bx2_0,bx2_1], dtype=tl.float32, is_pure=True, pack=1)
    overall_expsum = tl.where((dims & 32) != 0, den_b, den_a)
    even = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s2,o2], dtype=tl.float32, is_pure=True, pack=1)
    even = tl.fma(s0, o0, even)
    even = tl.fma(s4, o4, even)
    even = tl.fma(s6, o6, even)
    even = tl.fma(s8, o8, even)
    even = tl.fma(s10, o10, even)
    even = tl.fma(s12, o12, even)
    even = tl.fma(s14, o14, even)
    odd = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[s3,o3], dtype=tl.float32, is_pure=True, pack=1)
    odd = tl.fma(s1, o1, odd)
    odd = tl.fma(s5, o5, odd)
    odd = tl.fma(s7, o7, odd)
    odd = tl.fma(s9, o9, odd)
    odd = tl.fma(s11, o11, odd)
    odd = tl.fma(s13, o13, odd)
    odd = tl.fma(s15, o15, odd)
    acc_sum = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[even,odd], dtype=tl.float32, is_pure=True, pack=1)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)
