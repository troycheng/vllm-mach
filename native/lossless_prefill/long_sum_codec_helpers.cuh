// SPDX-License-Identifier: Apache-2.0
// SUM-only signed-zero extension of the frozen compact codec.  Input packing
// continues to use codec_helpers.cuh unchanged.  bit8 is raw and bit9 is a
// compact signed-zero mode; all other header/code bits retain their old roles.
__device__ __forceinline__ void store_packed_sum(vec_t<T, 8> const& value,
                                                  void* buffer, int idx,
                                                  int header_offset) {
  constexpr unsigned kRaw = 0x100;
  constexpr unsigned kZeroMode = 0x200;
  int block = idx / 32, lane = idx % 32;
  auto* payload = reinterpret_cast<uint4*>(buffer);
  auto* headers = reinterpret_cast<unsigned short*>(
      reinterpret_cast<unsigned char*>(buffer) + header_offset);
  unsigned mask = __activemask();
  alignas(16) unsigned short bits[8];
  // Zero has exponent 0, so max(all exponent) is also max(nonzero exponent).
  // A min initialized to 255 records whether any nonzero lane exists.
  unsigned hi = 0, lo_nonzero = 255, flags = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    bits[i] = __bfloat16_as_ushort(value[i]);
    unsigned exp = (bits[i] >> 7) & 255;
    bool zero = (bits[i] & 32767) == 0;
    hi = max(hi, exp);
    if (zero) {
      flags |= 1;  // any signed zero
    } else {
      lo_nonzero = min(lo_nonzero, exp);
    }
    if (exp == 255) flags |= 2;  // NaN/Inf
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    hi = max(hi, __shfl_xor_sync(mask, hi, offset, 32));
    lo_nonzero = min(lo_nonzero, __shfl_xor_sync(mask, lo_nonzero, offset, 32));
    flags |= __shfl_xor_sync(mask, flags, offset, 32);
  }
  bool zero_mode = flags & 1;
  // `lo_nonzero == 255` denotes all signed zeros unless bit1 already forces
  // raw for a special value.  hi remains the original max exponent for headers.
  bool raw = (flags & 2) ||
      (lo_nonzero != 255 && hi - lo_nonzero > (zero_mode ? 14 : 15));
  if (lane == 0) {
    headers[block] = static_cast<unsigned short>(hi | (raw ? kRaw : 0) |
                                                  (!raw && zero_mode ? kZeroMode : 0));
  }
  if (raw) {
    payload[idx] = *reinterpret_cast<uint4 const*>(bits);
    return;
  }
  mask = __activemask();
  unsigned code[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    bool zero = (bits[i] & 32767) == 0;
    unsigned exp = (bits[i] >> 7) & 255;
    unsigned delta = zero_mode && zero ? 15 : hi - exp;
    code[i] = (bits[i] & 127) | ((bits[i] >> 8) & 128) | (delta << 8);
  }
  unsigned w0 = code[0] | (code[1] << 12) | (code[2] << 24);
  unsigned w1 = (code[2] >> 8) | (code[3] << 4) | (code[4] << 16) | (code[5] << 28);
  unsigned w2 = (code[5] >> 4) | (code[6] << 8) | (code[7] << 20);
  uint4 wire;
  wire.x = pull3(w0, w1, w2, lane * 4, mask);
  wire.y = pull3(w0, w1, w2, lane * 4 + 1, mask);
  wire.z = pull3(w0, w1, w2, lane * 4 + 2, mask);
  wire.w = pull3(w0, w1, w2, lane * 4 + 3, mask);
  if (lane < 24) payload[block * 32 + lane] = wire;
}
