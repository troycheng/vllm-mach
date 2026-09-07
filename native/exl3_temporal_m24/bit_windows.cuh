// Exact decoder alternative. For the existing GEMM mapping idx is a multiple
// of four. No trellis repack, lookup table, or codebook arithmetic change.
#pragma once

template<int Bits, int Cb>
__device__ __forceinline__ void dq4_window(const uint32_t* ptr, int idx, FragB& frag) {
    static_assert(Bits == 5 || Bits == 6);
    int start = (idx + 257) * Bits - 16;
    int end = start + 3 * Bits + 16;
    int i0 = start / 32, i1 = (end - 1) / 32;
    int shift = (i1 + 1) * 32 - end;
    uint32_t a = ptr[i0 % (Bits * 8)], b = ptr[i1 % (Bits * 8)];
    uint32_t low = __funnelshift_r(b, a, shift);
    uint32_t w3 = low & 65535u;
    uint32_t w2 = (low >> Bits) & 65535u;
    uint32_t w1 = (low >> (2 * Bits)) & 65535u;
    uint32_t w0;
    if constexpr (Bits == 5) {
        // Four K5 states occupy 16 + 3*5 = 31 bits.
        w0 = (low >> 15) & 65535u;
    } else {
        // Four K6 states occupy 34 bits. idx % 4 == 0 guarantees
        // shift in {0,8,16,24}; shift+2 therefore never wraps at 32.
        uint32_t high = __funnelshift_r(b, a, shift + 2);
        w0 = high >> 16;
    }
    frag[0] = decode_3inst_2<Cb>(w0, w1);
    frag[1] = decode_3inst_2<Cb>(w2, w3);
}

template<int Bits, int Cb>
__device__ __forceinline__ void dq_dispatch_window(const uint32_t* ptr, int idx,
                                                   FragB& f0, FragB& f1) {
    dq4_window<Bits, Cb>(ptr, idx, f0);
    dq4_window<Bits, Cb>(ptr, idx + 4, f1);
}
