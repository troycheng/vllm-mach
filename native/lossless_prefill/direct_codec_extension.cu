// SPDX-License-Identifier: Apache-2.0
// TP2 BF16 AllReduce + residual + Gemma RMSNorm direct-own-half experiment.
// variant=1 only: peer-half input pack is inlined before barrier 1 in each CTA.
// FP32 reduction, BF16 conversion, aliasing, barriers and PDL are otherwise
// copied from the frozen half-pack candidate unchanged.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "trtllm_allreduce_fusion.cuh"

namespace mach_lossless_prefill_direct {
using namespace flashinfer::trtllm_allreduce_fusion;
using flashinfer::vec_t;
using T = __nv_bfloat16;
constexpr auto Pattern = AllReduceFusionPattern::kARResidualRMSNorm;
#include "direct_codec_helpers.cuh"
#include "direct_sum_codec_helpers.cuh"

constexpr int kM = 4096;
constexpr int kH = 5120;
constexpr int kVecSize = 8;
constexpr int kRanks = 2;
constexpr int kInputElements = kM * kH;
constexpr int kInputPayloadBytes = kInputElements * int(sizeof(T));
constexpr int kInputHeaderBytes = kInputElements / 256 * int(sizeof(unsigned short));
constexpr int kInputHeaderOffset = kInputPayloadBytes * 2;
constexpr int kSumPayloadOffset = kInputPayloadBytes;
constexpr int kSumHeaderOffset = kInputHeaderOffset + kInputHeaderBytes;
constexpr int kRequiredWorkspaceBytes =
    kInputElements * int(sizeof(T)) * 2 +
    kInputElements / 256 * int(sizeof(unsigned short)) * 2;
static_assert(kInputHeaderOffset == 83886080);
static_assert(kSumHeaderOffset == 84049920);
static_assert(kRequiredWorkspaceBytes == 84213760);

// Input format remains the frozen codec: bit9 is intentionally ignored because
// input packing still routes any signed-zero block through raw fallback.
__device__ __forceinline__ vec_t<T, 8> decode_peer_input(void const* buffer, int idx,
                                                           int header_offset) {
  auto const* payload = reinterpret_cast<uint4 const*>(buffer);
  auto const* headers = reinterpret_cast<unsigned short const*>(
      reinterpret_cast<unsigned char const*>(buffer) + header_offset);
  int block = idx / 32, lane = idx % 32;
  unsigned mask = __activemask();
  unsigned header = headers[block];
  vec_t<T, 8> result;
  if (header & 256) {
    result.load(reinterpret_cast<T const*>(buffer) + idx * 8);
    return result;
  }
  uint4 wire = {0, 0, 0, 0};
  if (lane < 24) wire = payload[block * 32 + lane];
  unsigned w0 = pull4(wire, lane * 3, mask);
  unsigned w1 = pull4(wire, lane * 3 + 1, mask);
  unsigned w2 = pull4(wire, lane * 3 + 2, mask);
  unsigned code[8] = {w0 & 4095, (w0 >> 12) & 4095,
      ((w0 >> 24) | (w1 << 8)) & 4095, (w1 >> 4) & 4095,
      (w1 >> 16) & 4095, ((w1 >> 28) | (w2 << 4)) & 4095,
      (w2 >> 8) & 4095, w2 >> 20};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    unsigned short bits = static_cast<unsigned short>((code[i] & 127) |
        ((code[i] & 128) << 8) | (((header & 255) - (code[i] >> 8)) << 7));
    result[i] = __ushort_as_bfloat16(bits);
  }
  return result;
}

// SUM format adds bit9.  In zero mode only, a 12-bit code whose delta is 15
// reconstructs a signed zero from its preserved sign code bit; other values
// use the frozen compact reconstruction unchanged.
__device__ __forceinline__ vec_t<T, 8> decode_peer_sum(void const* buffer, int idx,
                                                         int header_offset) {
  constexpr unsigned kRaw = 0x100;
  constexpr unsigned kZeroMode = 0x200;
  auto const* payload = reinterpret_cast<uint4 const*>(buffer);
  auto const* headers = reinterpret_cast<unsigned short const*>(
      reinterpret_cast<unsigned char const*>(buffer) + header_offset);
  int block = idx / 32, lane = idx % 32;
  unsigned mask = __activemask();
  unsigned header = headers[block];
  vec_t<T, 8> result;
  if (header & kRaw) {
    result.load(reinterpret_cast<T const*>(buffer) + idx * 8);
    return result;
  }
  uint4 wire = {0, 0, 0, 0};
  if (lane < 24) wire = payload[block * 32 + lane];
  unsigned w0 = pull4(wire, lane * 3, mask);
  unsigned w1 = pull4(wire, lane * 3 + 1, mask);
  unsigned w2 = pull4(wire, lane * 3 + 2, mask);
  unsigned code[8] = {w0 & 4095, (w0 >> 12) & 4095,
      ((w0 >> 24) | (w1 << 8)) & 4095, (w1 >> 4) & 4095,
      (w1 >> 16) & 4095, ((w1 >> 28) | (w2 << 4)) & 4095,
      (w2 >> 8) & 4095, w2 >> 20};
  bool zero_mode = header & kZeroMode;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    unsigned delta = code[i] >> 8;
    unsigned short bits;
    if (zero_mode && delta == 15) {
      // Compact zero code has mantissa bits zero; retain only BF16 sign.
      bits = static_cast<unsigned short>((code[i] & 128) << 8);
    } else {
      bits = static_cast<unsigned short>((code[i] & 127) |
          ((code[i] & 128) << 8) | (((header & 255) - delta) << 7));
    }
    result[i] = __ushort_as_bfloat16(bits);
  }
  return result;
}

// IndexHelper maps CTA c to token c + q*gridDim.x.  The inline input
// pack publishes the peer-consumed half before barrier 1.  CTA c later writes
// the SUM for its own half to peer CTA c, which consumes it after barrier 2.
__global__ void packed_twoshot_direct(AllReduceFusionParams<T> params) {
  constexpr int VEC_SIZE = 8, NRanks = 2;
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  // Direct execution consumes own half first.  Seed FusedOp's constructor
  // preload with that first access so rank 1 does not read lower-half residual
  // before update() switches to its upper-half first token.
  int own_access_base = params.rank * params.size / NRanks / VEC_SIZE;
  FusedOp<Pattern, T> fused_op(params, access_id + own_access_base, access_id_in_token);
  cudaGridDependencySynchronize();
  SyncComm<NRanks> comm(params.workspace);
  int half_tokens = params.size / params.hidden_dim / NRanks;
  int header_offset = params.size * sizeof(T) * 2;

  // Variant 1 only: this rank packs the input half peer CTA c will decode.
  int peer_begin = (1 - params.rank) * half_tokens;
  int peer_access_id = access_id + peer_begin * params.hidden_dim / VEC_SIZE;
  int peer_tot_access = (peer_begin + half_tokens) * params.hidden_dim / VEC_SIZE;
  for (int idx = peer_access_id; idx < peer_tot_access; idx += access_stride) {
    uint4 original = reinterpret_cast<uint4 const*>(params.allreduce_in)[idx];
    store_packed_input(original, comm.comm_bufs[params.rank], idx, header_offset);
  }

  Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();
  int own_begin = params.rank * half_tokens;
  int own_access_id = access_id + own_begin * params.hidden_dim / VEC_SIZE;
  int own_tot_access = (own_begin + half_tokens) * params.hidden_dim / VEC_SIZE;
  int sum_header_relative = params.size * sizeof(T) +
      params.size / 256 * sizeof(unsigned short);
  // The allreduce and BF16 sum are unchanged.  Publish compressed SUM to the
  // peer, then immediately consume the same local value; no local raw SUM
  // payload/header is written for this own half.
  for (int idx = own_access_id, tidx = token_id + own_begin; idx < own_tot_access;
       idx += access_stride, tidx += token_stride) {
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      if (r == params.rank)
        vals[r].load(reinterpret_cast<T const*>(params.allreduce_in) + idx * VEC_SIZE);
      else
        vals[r] = decode_peer_input(comm.comm_bufs[r], idx, header_offset);
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, true>(vals);
    store_packed_sum(sum_val,
        reinterpret_cast<T*>(comm.comm_bufs[1 - params.rank]) + tot_access * VEC_SIZE,
        idx, sum_header_relative);
    fused_op.update(idx);
    fused_op(sum_val, tidx);
  }
  // Same system-scope publication protocol: this makes peer SUM payload and
  // header visible before the peer-half decoder below starts.
  barrier.sync();

  int remote_begin = (1 - params.rank) * half_tokens;
  int remote_access_id = access_id + remote_begin * params.hidden_dim / VEC_SIZE;
  int remote_tot_access = (remote_begin + half_tokens) * params.hidden_dim / VEC_SIZE;
  for (int idx = remote_access_id, tidx = token_id + remote_begin; idx < remote_tot_access;
       idx += access_stride, tidx += token_stride) {
    fused_op.update(idx);
    vec_t<T, VEC_SIZE> sum_val = decode_peer_sum(
        reinterpret_cast<T*>(comm.comm_bufs[params.rank]) + tot_access * VEC_SIZE,
        idx, sum_header_relative);
    fused_op(sum_val, tidx);
  }
  comm.update(barrier.m_flag_value);
  cudaTriggerProgrammaticLaunchCompletion();
}

// Test-only: never called by run().  It uses the device-resident pointer table
// through workspace[rank], so host code never dereferences a GPU pointer table.
__global__ void debug_unused_input_half_kernel(void** workspace, int rank,
                                                uint32_t pattern, bool check,
                                                int* errors) {
  constexpr int kHalfPayloadBytes = kInputElements * int(sizeof(T)) / 2;
  constexpr int kHalfHeaderBytes = (kInputElements / 256) * int(sizeof(unsigned short)) / 2;
  constexpr int kPayloadWords = kHalfPayloadBytes / int(sizeof(uint32_t));
  constexpr int kHeaderWords = kHalfHeaderBytes / int(sizeof(uint32_t));
  static_assert(kInputHeaderOffset == 83886080);
  static_assert(kHalfPayloadBytes == 20 * 1024 * 1024);
  static_assert(kHalfHeaderBytes == 81920);
  // r0 produces upper for r1, leaving lower unused; r1 is the converse.
  int unused_half = rank;
  auto* base = reinterpret_cast<unsigned char*>(workspace[rank]);
  auto* payload = reinterpret_cast<uint32_t*>(base) + unused_half * kPayloadWords;
  auto* headers = reinterpret_cast<uint32_t*>(base + kInputHeaderOffset) +
      unused_half * kHeaderWords;
  for (int i = int(blockIdx.x * blockDim.x + threadIdx.x);
       i < kPayloadWords + kHeaderWords; i += int(blockDim.x * gridDim.x)) {
    uint32_t* addr = i < kPayloadWords ? payload + i : headers + (i - kPayloadWords);
    if (check) {
      if (*addr != pattern) atomicAdd(errors, 1);
    } else {
      *addr = pattern;
    }
  }
}

// Test-only: the direct path no longer writes or reads this rank's own
// SUM half.  Its peer-owned half remains the active received-SUM region.
__global__ void debug_unused_sum_half_kernel(void** workspace, int rank,
                                              uint32_t pattern, bool check,
                                              int* errors) {
  constexpr int kHalfPayloadBytes = kInputPayloadBytes / 2;
  constexpr int kHalfHeaderBytes = kInputHeaderBytes / 2;
  constexpr int kPayloadWords = kHalfPayloadBytes / int(sizeof(uint32_t));
  constexpr int kHeaderWords = kHalfHeaderBytes / int(sizeof(uint32_t));
  static_assert(kSumPayloadOffset == 41943040);
  static_assert(kSumHeaderOffset == 84049920);
  static_assert(kHalfPayloadBytes == 20 * 1024 * 1024);
  static_assert(kHalfHeaderBytes == 81920);
  auto* base = reinterpret_cast<unsigned char*>(workspace[rank]);
  auto* payload = reinterpret_cast<uint32_t*>(base + kSumPayloadOffset) +
      rank * kPayloadWords;
  auto* headers = reinterpret_cast<uint32_t*>(base + kSumHeaderOffset) +
      rank * kHeaderWords;
  for (int i = int(blockIdx.x * blockDim.x + threadIdx.x);
       i < kPayloadWords + kHeaderWords; i += int(blockDim.x * gridDim.x)) {
    uint32_t* addr = i < kPayloadWords ? payload + i : headers + (i - kPayloadWords);
    if (check) {
      if (*addr != pattern) atomicAdd(errors, 1);
    } else {
      *addr = pattern;
    }
  }
}

// Roundtrip is test-only and deliberately does not share workspace or call run.
// wire layout: [num_groups * 512 B raw-sized payload][num_groups * 2 B header].
__global__ void codec_roundtrip_store_kernel(T const* input, unsigned char* wire,
                                             int total_vectors, int header_offset) {
  int idx = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_vectors) return;
  vec_t<T, 8> value;
  value.load(input + idx * 8);
  store_packed_sum(value, wire, idx, header_offset);
}

__global__ void codec_roundtrip_decode_kernel(unsigned char const* wire, T* output,
                                              unsigned short* observed_headers,
                                              int total_vectors, int header_offset) {
  int idx = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_vectors) return;
  vec_t<T, 8> value = decode_peer_sum(wire, idx, header_offset);
  value.store(output + idx * 8);
  if (idx % 32 == 0) {
    observed_headers[idx / 32] = reinterpret_cast<unsigned short const*>(wire + header_offset)[idx / 32];
  }
}

void codec_roundtrip(at::Tensor const& input, at::Tensor const& wire,
                     at::Tensor const& output, at::Tensor& headers) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 1 && input.is_contiguous() &&
              input.scalar_type() == at::kBFloat16 && input.numel() > 0 && input.numel() % 256 == 0);
  TORCH_CHECK(output.device() == input.device() && output.is_contiguous() &&
              output.scalar_type() == at::kBFloat16 && output.sizes() == input.sizes());
  TORCH_CHECK(wire.device() == input.device() && wire.is_contiguous() && wire.scalar_type() == at::kByte &&
              reinterpret_cast<uintptr_t>(wire.data_ptr()) % 16 == 0);
  int64_t groups = input.numel() / 256;
  int64_t payload_bytes = input.numel() * int64_t(sizeof(T));
  int64_t header_bytes = groups * int64_t(sizeof(unsigned short));
  TORCH_CHECK(wire.numel() >= payload_bytes + header_bytes,
              "wire needs raw-sized payload plus header tail");
  TORCH_CHECK(headers.device() == input.device() && headers.is_contiguous() &&
              headers.scalar_type() == at::kUInt16 && headers.numel() == groups);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0);
  c10::cuda::CUDAGuard guard(input.device());
  int total_vectors = int(input.numel() / kVecSize);
  int blocks = (total_vectors + 255) / 256;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  codec_roundtrip_store_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<T const*>(input.data_ptr()),
      reinterpret_cast<unsigned char*>(wire.data_ptr()), total_vectors, int(payload_bytes));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "codec roundtrip store launch failed");
  codec_roundtrip_decode_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<unsigned char const*>(wire.data_ptr()),
      reinterpret_cast<T*>(output.data_ptr()), headers.data_ptr<unsigned short>(),
      total_vectors, int(payload_bytes));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "codec roundtrip decode launch failed");
}

void run(at::Tensor const& input, at::Tensor const& residual,
         at::Tensor const& gamma, at::Tensor const& workspace,
         at::Tensor const& residual_out, at::Tensor const& norm_out,
         int64_t rank, int64_t workspace_bytes, double eps, double weight_bias,
         int64_t variant, bool pdl) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(0) == kM && input.size(1) == kH);
  for (auto const& x : {input, residual, gamma, residual_out, norm_out}) {
    TORCH_CHECK(x.device() == input.device() && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(residual.sizes() == input.sizes() && residual_out.sizes() == input.sizes() && norm_out.sizes() == input.sizes());
  TORCH_CHECK(gamma.numel() == kH && rank >= 0 && rank < kRanks && eps > 0);
  TORCH_CHECK(variant == 1, "mach_lossless_prefill_direct supports only inline half-pack variant=1");
  TORCH_CHECK(workspace.device() == input.device() && workspace.is_contiguous() &&
              workspace.scalar_type() == at::kLong && workspace.numel() >= 7);
  TORCH_CHECK(workspace_bytes >= kRequiredWorkspaceBytes);
  c10::cuda::CUDAGuard guard(input.device());
  AllReduceFusionParams<T> params{};
  params.nranks = kRanks; params.rank = int(rank);
  params.size = int(input.numel()); params.hidden_dim = kH;
  params.workspace = reinterpret_cast<void**>(workspace.data_ptr());
  params.allreduce_in = input.data_ptr(); params.residual_in = residual.data_ptr();
  params.residual_out = residual_out.data_ptr(); params.norm_out = norm_out.data_ptr();
  params.rms_gamma = gamma.data_ptr(); params.rms_eps = float(eps);
  params.weight_bias = float(weight_bias); params.use_oneshot = false;
  params.stream = c10::cuda::getCurrentCUDAStream().stream();
  params.pattern = Pattern; params.trigger_completion_at_end = true;

  int active = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, packed_twoshot_direct, 640, 0) == cudaSuccess && active >= 1,
      "direct kernel cannot sustain the original 640-thread whole-token block");
  int sm_count = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, input.get_device()) == cudaSuccess);
  cudaLaunchConfig_t config{};
  config.gridDim = sm_count; config.blockDim = 640; config.stream = params.stream;
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attributes[1].id = cudaLaunchAttributeClusterDimension;
  attributes[1].val.clusterDim.x = 1;
  attributes[1].val.clusterDim.y = 1;
  attributes[1].val.clusterDim.z = 1;
  config.attrs = attributes; config.numAttrs = 2;
  cudaError_t error = cudaLaunchKernelEx(&config, packed_twoshot_direct, params);
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

void debug_unused_input_half(at::Tensor const& workspace, int64_t rank,
                             int64_t workspace_bytes, int64_t pattern,
                             bool check, at::Tensor& errors) {
  TORCH_CHECK(rank >= 0 && rank < kRanks && workspace_bytes >= kRequiredWorkspaceBytes);
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.scalar_type() == at::kLong &&
              workspace.numel() >= 7);
  TORCH_CHECK(errors.is_cuda() && errors.device() == workspace.device() && errors.is_contiguous() &&
              errors.scalar_type() == at::kInt && errors.numel() == 1);
  c10::cuda::CUDAGuard guard(workspace.device());
  constexpr int kThreads = 256;
  constexpr int kHalfPayloadWords = (kInputElements * int(sizeof(T)) / 2) / int(sizeof(uint32_t));
  constexpr int kHalfHeaderWords = ((kInputElements / 256) * int(sizeof(unsigned short)) / 2) / int(sizeof(uint32_t));
  int blocks = (kHalfPayloadWords + kHalfHeaderWords + kThreads - 1) / kThreads;
  debug_unused_input_half_kernel<<<blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<void**>(workspace.data_ptr()), int(rank), uint32_t(pattern), check,
      errors.data_ptr<int>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "debug unused-input-half launch failed");
}

void debug_unused_sum_half(at::Tensor const& workspace, int64_t rank,
                           int64_t workspace_bytes, int64_t pattern,
                           bool check, at::Tensor& errors) {
  TORCH_CHECK(rank >= 0 && rank < kRanks && workspace_bytes >= kRequiredWorkspaceBytes);
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.scalar_type() == at::kLong &&
              workspace.numel() >= 7);
  TORCH_CHECK(errors.is_cuda() && errors.device() == workspace.device() && errors.is_contiguous() &&
              errors.scalar_type() == at::kInt && errors.numel() == 1);
  c10::cuda::CUDAGuard guard(workspace.device());
  constexpr int kThreads = 256;
  constexpr int kHalfPayloadWords = (kInputPayloadBytes / 2) / int(sizeof(uint32_t));
  constexpr int kHalfHeaderWords = (kInputHeaderBytes / 2) / int(sizeof(uint32_t));
  int blocks = (kHalfPayloadWords + kHalfHeaderWords + kThreads - 1) / kThreads;
  debug_unused_sum_half_kernel<<<blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<void**>(workspace.data_ptr()), int(rank), uint32_t(pattern), check,
      errors.data_ptr<int>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "debug unused-sum-half launch failed");
}

std::vector<int64_t> info(at::Tensor const& input, int64_t variant) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(0) == kM && input.size(1) == kH);
  TORCH_CHECK(variant == 1, "mach_lossless_prefill_direct supports only variant=1");
  c10::cuda::CUDAGuard guard(input.device());
  cudaFuncAttributes candidate{}, control{};
  TORCH_CHECK(cudaFuncGetAttributes(&candidate, packed_twoshot_direct) == cudaSuccess);
  TORCH_CHECK(cudaFuncGetAttributes(&control,
      allreduce_fusion_kernel_twoshot_sync<Pattern, T, 2, true>) == cudaSuccess);
  int active = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, packed_twoshot_direct, 640, 0) == cudaSuccess);
  // [variant, candidate regs, stack, shared, control regs, stack, shared, active blocks/SM]
  return {variant, candidate.numRegs, int64_t(candidate.localSizeBytes),
          int64_t(candidate.sharedSizeBytes), control.numRegs,
          int64_t(control.localSizeBytes), int64_t(control.sharedSizeBytes), active};
}
}  // namespace mach_lossless_prefill_direct

TORCH_LIBRARY(mach_lossless_prefill_direct, m) {
  m.def("run(Tensor input, Tensor residual, Tensor gamma, Tensor workspace, Tensor(a!) residual_out, Tensor(b!) norm_out, int rank, int workspace_bytes, float eps, float weight_bias, int variant, bool pdl) -> ()");
  m.def("debug_unused_input_half(Tensor workspace, int rank, int workspace_bytes, int pattern, bool check, Tensor(a!) errors) -> ()");
  m.def("debug_unused_sum_half(Tensor workspace, int rank, int workspace_bytes, int pattern, bool check, Tensor(a!) errors) -> ()");
  m.def("codec_roundtrip(Tensor input, Tensor wire, Tensor(a!) output, Tensor(b!) headers) -> ()");
  m.def("info(Tensor input, int variant) -> int[]");
}
TORCH_LIBRARY_IMPL(mach_lossless_prefill_direct, CUDA, m) {
  m.impl("run", &mach_lossless_prefill_direct::run);
  m.impl("debug_unused_input_half", &mach_lossless_prefill_direct::debug_unused_input_half);
  m.impl("debug_unused_sum_half", &mach_lossless_prefill_direct::debug_unused_sum_half);
  m.impl("codec_roundtrip", &mach_lossless_prefill_direct::codec_roundtrip);
  m.impl("info", &mach_lossless_prefill_direct::info);
}
