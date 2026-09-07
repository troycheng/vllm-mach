// SPDX-License-Identifier: Apache-2.0
// One full warp owns one aligned 256-value block. All current token, rank and
// loop boundaries are multiples of 32 vectors. buffer points at the SUM region;
// header_offset is relative to that pointer, after both payload regions and
// the separate input header array. Original cross-GPU barriers publish stores.
__device__ __forceinline__ void store_packed_sum(vec_t<T, 8> const& value,
                                                void* buffer, int idx,
                                                int header_offset) {
  int block = idx / 32, lane = idx % 32;
  auto* payload = reinterpret_cast<uint4*>(buffer);
  auto* headers = reinterpret_cast<unsigned short*>(
      reinterpret_cast<unsigned char*>(buffer) + header_offset);
  unsigned mask = __activemask();
  alignas(16) unsigned short bits[8];
  unsigned lo = 255, hi = 0, bad = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    bits[i] = __bfloat16_as_ushort(value[i]);
    unsigned e = (bits[i] >> 7) & 255;
    lo = min(lo, e); hi = max(hi, e);
    bad |= e == 255 || (bits[i] & 32767) == 0;
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    lo = min(lo, __shfl_xor_sync(mask, lo, offset, 32));
    hi = max(hi, __shfl_xor_sync(mask, hi, offset, 32));
    bad |= __shfl_xor_sync(mask, bad, offset, 32);
  }
  bool raw = bad || hi - lo > 15;
  // This narrow remote store is a measured cost, not assumed free. Publishing
  // it before the payload is safe because no consumer runs before barrier 2.
  if (lane == 0) headers[block] = static_cast<unsigned short>(hi | (raw ? 256 : 0));
  if (raw) {
    payload[idx] = *reinterpret_cast<uint4 const*>(bits);
    return;
  }
  unsigned code[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    code[i] = (bits[i] & 127) | ((bits[i] >> 8) & 128) |
              ((hi - ((bits[i] >> 7) & 255)) << 8);
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
