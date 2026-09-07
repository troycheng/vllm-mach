// SPDX-License-Identifier: Apache-2.0
// Exact BF16 input codec helpers.  One full warp owns one aligned 256-value
// block.  The caller supplies a global vector idx, so payload/header locations
// remain global even when only a token half is packed.
__device__ __forceinline__ unsigned pull3(unsigned a, unsigned b, unsigned c,
                                          int word, unsigned mask) {
  int source = min(word / 3, 31);
  unsigned aa = __shfl_sync(mask, a, source, 32);
  unsigned bb = __shfl_sync(mask, b, source, 32);
  unsigned cc = __shfl_sync(mask, c, source, 32);
  return word % 3 == 0 ? aa : word % 3 == 1 ? bb : cc;
}

__device__ __forceinline__ unsigned pull4(uint4 v, int word, unsigned mask) {
  int source = word / 4;
  unsigned a = __shfl_sync(mask, v.x, source, 32);
  unsigned b = __shfl_sync(mask, v.y, source, 32);
  unsigned c = __shfl_sync(mask, v.z, source, 32);
  unsigned d = __shfl_sync(mask, v.w, source, 32);
  return word % 4 == 0 ? a : word % 4 == 1 ? b : word % 4 == 2 ? c : d;
}

// `idx` is a global vector index and is warp-aligned.  `buffer` starts at the
// input payload region; `header_offset` follows the two raw-sized payload
// regions.  This is byte-for-byte the former pack_workspace encoding.
__device__ __forceinline__ void store_packed_input(uint4 original, void* buffer,
                                                     int idx, int header_offset) {
  int block = idx / 32, lane = idx % 32;
  auto* payload = reinterpret_cast<uint4*>(buffer);
  auto* headers = reinterpret_cast<unsigned short*>(
      reinterpret_cast<unsigned char*>(buffer) + header_offset);
  unsigned mask = __activemask();
  alignas(16) unsigned short bits[8];
  *reinterpret_cast<uint4*>(bits) = original;
  unsigned lo = 255, hi = 0, bad = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
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
  if (lane == 0) headers[block] = static_cast<unsigned short>(hi | (raw ? 256 : 0));
  if (raw) {
    payload[idx] = original;
    return;
  }
  // Retain the original pack helper's post-branch full-warp mask refresh.
  mask = __activemask();
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

// Independent variant: a producer rank packs only the global token half the
// peer rank will decode.  The global block index deliberately is not compacted.
__global__ void pack_workspace_peer_half(uint4 const* input, void** workspace,
                                         int rank, int header_offset,
                                         int block_base, int half_blocks) {
  int local_thread = int(blockIdx.x * blockDim.x + threadIdx.x);
  int local_block = local_thread / 32;
  if (local_block >= half_blocks) return;
  int lane = local_thread % 32;
  int global_idx = (block_base + local_block) * 32 + lane;
  store_packed_input(input[global_idx], workspace[rank], global_idx, header_offset);
}
