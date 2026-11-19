def triton_per_fused__to_copy_add_mean_mul_mxfp8_quantize_pow_rsqrt_silu_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (((r0_1 + 128*x0) % (2048*ks0))), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (8192*(x0 // 16) + (((r0_1 + 128*x0) % 2048))), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 128.0, tl.float32)
    tmp10 = (tmp6 / tmp9)
    tmp11 = tl.full([1, 1], 1e-06, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp8 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = -tmp19
    tmp21 = libdevice.exp(tmp20)
    tmp22 = tl.full([1, 1], 1.0, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = (tmp19 / tmp23)
    tmp25 = tmp17 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (((r0_1 + 128*x0) % (2048*ks0))), tmp26, xmask)
