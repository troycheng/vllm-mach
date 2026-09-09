// SPDX-License-Identifier: Apache-2.0
// Isolated dual-load observed-M extension of the validated TP2 direct SUM codec.
// Isolated from frozen M4096 direct and normalized_twoshot_draft.  No service use.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "trtllm_allreduce_fusion.cuh"

namespace mach_lossless_prefill_long {
using namespace flashinfer::trtllm_allreduce_fusion;
using flashinfer::vec_t;
using T = __nv_bfloat16;
constexpr auto Pattern = AllReduceFusionPattern::kARResidualRMSNorm;
#include "long_codec_helpers.cuh"
#include "long_sum_codec_helpers.cuh"

constexpr int kH = 5120;
constexpr int kRanks = 2;
constexpr int kVecSize = 8;
constexpr int kThreads = 640;
constexpr int kCodecBlockElements = 256;
static_assert(kH / kVecSize == kThreads);

// This is deliberately the complete observed-M set, not a numeric interval.
__host__ __device__ constexpr bool is_supported_m(int m) {
  return m == 1024 || m == 1052 || m == 3000 || m == 3001 || m == 3002 || m == 3003 || m == 3012 || m == 3013 || m == 3014 || m == 3015 || m == 3020 || m == 3021 || m == 3022 || m == 3023 || m == 3028 || m == 3029 || m == 3030 || m == 3031 || m == 3093 || m == 3094 || m == 3095 || m == 3185 || m == 3244 || m == 3658 || m == 3842 || m == 3845 || m == 3866 || m == 3869 || m == 3890 || m == 3891 || m == 3893 || m == 3895;
}
__host__ __device__ constexpr bool stock_is_oneshot(int m) { return m <= 3276; }
__host__ __device__ constexpr int64_t workspace_bytes_for(int64_t m) {
  return m * kH * int64_t(sizeof(T)) * kRanks +
      (m * kH / kCodecBlockElements) * int64_t(sizeof(unsigned short)) * kRanks;
}
static_assert(workspace_bytes_for(3028) == 62255680);
static_assert(workspace_bytes_for(3895) == 80081200);

struct TokenPartition { int own_begin, own_tokens, peer_begin, peer_tokens; };
// Rank 0 owns ceil(M/2); rank 1 owns floor(M/2), including each observed odd M.
__host__ __device__ __forceinline__ TokenPartition partition_for(int m, int rank) {
  int lower = m / 2, upper = m - lower;
  return rank == 0 ? TokenPartition{0, upper, upper, lower}
                   : TokenPartition{upper, lower, 0, upper};
}

// Identical primitive to upstream one-shot's input normalization.  It is only
// instantiated only for stock-one-shot M<=3276, before peer codec publication.
__device__ __forceinline__ uint4 normalize_input_for_peer(uint4 original) {
  alignas(16) T before[kVecSize];
  *reinterpret_cast<uint4*>(before) = original;
  vec_t<T, kVecSize> value;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) value[i] = before[i];
  remove_neg_zero<T, kVecSize>(value);
  alignas(16) T after[kVecSize];
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) after[i] = value[i];
  return *reinterpret_cast<uint4 const*>(after);
}

// Input decode has no normalizer: for small M the producer already normalized;
// for large M it must retain the stock two-shot input bit pattern, including -0.
__device__ __forceinline__ vec_t<T, kVecSize> decode_peer_input(void const* buffer,
                                                                  int idx, int header_offset) {
  auto const* payload = reinterpret_cast<uint4 const*>(buffer);
  auto const* headers = reinterpret_cast<unsigned short const*>(
      reinterpret_cast<unsigned char const*>(buffer) + header_offset);
  int block = idx / 32, lane = idx % 32;
  unsigned mask = __activemask(), header = headers[block];
  vec_t<T, kVecSize> result;
  if (header & 0x100) {
    result.load(reinterpret_cast<T const*>(buffer) + idx * kVecSize);
    return result;
  }
  uint4 wire = {0, 0, 0, 0};
  if (lane < 24) wire = payload[block * 32 + lane];
  unsigned w0 = pull4(wire, lane * 3, mask);
  unsigned w1 = pull4(wire, lane * 3 + 1, mask);
  unsigned w2 = pull4(wire, lane * 3 + 2, mask);
  unsigned code[kVecSize] = {w0 & 4095, (w0 >> 12) & 4095,
      ((w0 >> 24) | (w1 << 8)) & 4095, (w1 >> 4) & 4095,
      (w1 >> 16) & 4095, ((w1 >> 28) | (w2 << 4)) & 4095,
      (w2 >> 8) & 4095, w2 >> 20};
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    unsigned short bits = static_cast<unsigned short>((code[i] & 127) |
        ((code[i] & 128) << 8) | (((header & 255) - (code[i] >> 8)) << 7));
    result[i] = __ushort_as_bfloat16(bits);
  }
  return result;
}

// Frozen direct SUM codec: raw bit8 and compact signed-zero bit9.
__device__ __forceinline__ vec_t<T, kVecSize> decode_peer_sum(void const* buffer,
                                                                int idx, int header_offset) {
  constexpr unsigned kRaw = 0x100, kZeroMode = 0x200;
  auto const* payload = reinterpret_cast<uint4 const*>(buffer);
  auto const* headers = reinterpret_cast<unsigned short const*>(
      reinterpret_cast<unsigned char const*>(buffer) + header_offset);
  int block = idx / 32, lane = idx % 32;
  unsigned mask = __activemask(), header = headers[block];
  vec_t<T, kVecSize> result;
  if (header & kRaw) {
    result.load(reinterpret_cast<T const*>(buffer) + idx * kVecSize);
    return result;
  }
  uint4 wire = {0, 0, 0, 0};
  if (lane < 24) wire = payload[block * 32 + lane];
  unsigned w0 = pull4(wire, lane * 3, mask);
  unsigned w1 = pull4(wire, lane * 3 + 1, mask);
  unsigned w2 = pull4(wire, lane * 3 + 2, mask);
  unsigned code[kVecSize] = {w0 & 4095, (w0 >> 12) & 4095,
      ((w0 >> 24) | (w1 << 8)) & 4095, (w1 >> 4) & 4095,
      (w1 >> 16) & 4095, ((w1 >> 28) | (w2 << 4)) & 4095,
      (w2 >> 8) & 4095, w2 >> 20};
  bool zero_mode = header & kZeroMode;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    unsigned delta = code[i] >> 8;
    unsigned short bits = zero_mode && delta == 15
        ? static_cast<unsigned short>((code[i] & 128) << 8)
        : static_cast<unsigned short>((code[i] & 127) | ((code[i] & 128) << 8) |
                                      (((header & 255) - delta) << 7));
    result[i] = __ushort_as_bfloat16(bits);
  }
  return result;
}

// Template choice is made by the host from the observed M dispatch: true only
// for allowed M<=3276 (stock one-shot); larger allowed M retain all BF16 bits.
template <bool NormalizeInput>
__global__ void packed_twoshot_direct(AllReduceFusionParams<T> params) {
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  int token_num = params.size / params.hidden_dim;
  TokenPartition partition = partition_for(token_num, params.rank);
  int own_access_base = partition.own_begin * params.hidden_dim / kVecSize;
  FusedOp<Pattern, T> fused_op(params, access_id + own_access_base, access_id_in_token);
  cudaGridDependencySynchronize();
  SyncComm<kRanks> comm(params.workspace);
  int input_payload_bytes = params.size * int(sizeof(T));
  int input_header_offset = input_payload_bytes * kRanks;
  int sum_header_relative = input_payload_bytes +
      params.size / kCodecBlockElements * int(sizeof(unsigned short));

  int peer_access_id = access_id + partition.peer_begin * params.hidden_dim / kVecSize;
  int peer_tot_access = (partition.peer_begin + partition.peer_tokens) *
      params.hidden_dim / kVecSize;
  for (int idx = peer_access_id; idx < peer_tot_access; idx += access_stride) {
    uint4 original = reinterpret_cast<uint4 const*>(params.allreduce_in)[idx];
    if constexpr (NormalizeInput) original = normalize_input_for_peer(original);
    store_packed_input(original, comm.comm_bufs[params.rank], idx, input_header_offset);
  }

  Barrier<kRanks> barrier(params.rank, comm);
  barrier.sync();
  int own_access_id = access_id + partition.own_begin * params.hidden_dim / kVecSize;
  int own_tot_access = (partition.own_begin + partition.own_tokens) *
      params.hidden_dim / kVecSize;
  for (int idx = own_access_id, tidx = token_id + partition.own_begin;
       idx < own_tot_access; idx += access_stride, tidx += token_stride) {
    vec_t<T, kVecSize> vals[kRanks];
#pragma unroll
    for (int r = 0; r < kRanks; ++r) {
      if (r == params.rank) {
        vals[r].load(reinterpret_cast<T const*>(params.allreduce_in) + idx * kVecSize);
        if constexpr (NormalizeInput) remove_neg_zero<T, kVecSize>(vals[r]);
      } else {
        vals[r] = decode_peer_input(comm.comm_bufs[r], idx, input_header_offset);
      }
    }
    // This preserves rank-0 then rank-1 FP32 addition and the BF16 store point.
    vec_t<T, kVecSize> sum_val = allreduce_sum<T, kVecSize, kRanks, true>(vals);
    store_packed_sum(sum_val,
        reinterpret_cast<T*>(comm.comm_bufs[1 - params.rank]) + tot_access * kVecSize,
        idx, sum_header_relative);
    fused_op.update(idx);
    fused_op(sum_val, tidx);
  }
  barrier.sync();

  int remote_access_id = access_id + partition.peer_begin * params.hidden_dim / kVecSize;
  int remote_tot_access = (partition.peer_begin + partition.peer_tokens) *
      params.hidden_dim / kVecSize;
  for (int idx = remote_access_id, tidx = token_id + partition.peer_begin;
       idx < remote_tot_access; idx += access_stride, tidx += token_stride) {
    fused_op.update(idx);
    vec_t<T, kVecSize> sum_val = decode_peer_sum(
        reinterpret_cast<T*>(comm.comm_bufs[params.rank]) + tot_access * kVecSize,
        idx, sum_header_relative);
    fused_op(sum_val, tidx);
  }
  comm.update(barrier.m_flag_value);
  cudaTriggerProgrammaticLaunchCompletion();
}

void validate_io(at::Tensor const& input, at::Tensor const& residual,
                 at::Tensor const& gamma, at::Tensor const& workspace,
                 at::Tensor const& residual_out, at::Tensor const& norm_out,
                 int64_t rank, int64_t workspace_bytes, double eps) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && is_supported_m(int(input.size(0))) &&
              input.size(1) == kH && input.scalar_type() == at::kBFloat16 && input.is_contiguous());
  for (auto const& x : {input, residual, gamma, residual_out, norm_out}) {
    TORCH_CHECK(x.device() == input.device() && x.is_contiguous() &&
                x.scalar_type() == at::kBFloat16);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(residual.sizes() == input.sizes() && residual_out.sizes() == input.sizes() &&
              norm_out.sizes() == input.sizes() && gamma.numel() == kH);
  TORCH_CHECK(rank >= 0 && rank < kRanks && eps > 0);
  TORCH_CHECK(workspace.device() == input.device() && workspace.is_contiguous() &&
              workspace.scalar_type() == at::kLong && workspace.numel() >= 7);
  TORCH_CHECK(workspace_bytes >= workspace_bytes_for(input.size(0)));
}

void run(at::Tensor const& input, at::Tensor const& residual,
         at::Tensor const& gamma, at::Tensor const& workspace,
         at::Tensor const& residual_out, at::Tensor const& norm_out,
         int64_t rank, int64_t workspace_bytes, double eps, double weight_bias,
         bool pdl) {
  validate_io(input, residual, gamma, workspace, residual_out, norm_out, rank,
              workspace_bytes, eps);
  c10::cuda::CUDAGuard guard(input.device());
  AllReduceFusionParams<T> params{};
  params.nranks = kRanks; params.rank = int(rank); params.size = int(input.numel());
  params.hidden_dim = kH; params.workspace = reinterpret_cast<void**>(workspace.data_ptr());
  params.allreduce_in = input.data_ptr(); params.residual_in = residual.data_ptr();
  params.residual_out = residual_out.data_ptr(); params.norm_out = norm_out.data_ptr();
  params.rms_gamma = gamma.data_ptr(); params.rms_eps = float(eps);
  params.weight_bias = float(weight_bias); params.use_oneshot = false;
  params.stream = c10::cuda::getCurrentCUDAStream().stream(); params.pattern = Pattern;
  params.trigger_completion_at_end = true;
  int active = 0;
  if (stock_is_oneshot(int(input.size(0)))) {
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, packed_twoshot_direct<true>, kThreads, 0) == cudaSuccess && active >= 1);
  } else {
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, packed_twoshot_direct<false>, kThreads, 0) == cudaSuccess && active >= 1);
  }
  int sm_count = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    input.get_device()) == cudaSuccess);
  cudaLaunchConfig_t config{};
  config.gridDim = sm_count; config.blockDim = kThreads; config.stream = params.stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = attrs[1].val.clusterDim.y = attrs[1].val.clusterDim.z = 1;
  config.attrs = attrs; config.numAttrs = 2;
  cudaError_t error = stock_is_oneshot(int(input.size(0)))
      ? cudaLaunchKernelEx(&config, packed_twoshot_direct<true>, params)
      : cudaLaunchKernelEx(&config, packed_twoshot_direct<false>, params);
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

// Call info once for every allowed input M. stock_mode 1 is
// the upstream one-shot control and 2 is upstream two-shot; properties are for
// the same 640-thread, cluster-1 request, while trace remains launch authority.
std::vector<int64_t> info(at::Tensor const& input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && is_supported_m(int(input.size(0))) &&
              input.size(1) == kH && input.scalar_type() == at::kBFloat16 && input.is_contiguous());
  c10::cuda::CUDAGuard guard(input.device());
  bool one = stock_is_oneshot(int(input.size(0)));
  cudaFuncAttributes candidate{}, control{};
  cudaError_t status = one
      ? cudaFuncGetAttributes(&candidate, packed_twoshot_direct<true>)
      : cudaFuncGetAttributes(&candidate, packed_twoshot_direct<false>);
  TORCH_CHECK(status == cudaSuccess);
  status = one
      ? cudaFuncGetAttributes(&control,
          allreduce_fusion_kernel_oneshot_lamport<Pattern, T, kRanks, true, true>)
      : cudaFuncGetAttributes(&control,
          allreduce_fusion_kernel_twoshot_sync<Pattern, T, kRanks, true>);
  TORCH_CHECK(status == cudaSuccess);
  int candidate_active = 0, control_active = 0, sm_count = 0;
  status = one
      ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &candidate_active, packed_twoshot_direct<true>, kThreads, 0)
      : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &candidate_active, packed_twoshot_direct<false>, kThreads, 0);
  TORCH_CHECK(status == cudaSuccess);
  status = one
      ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &control_active,
          allreduce_fusion_kernel_oneshot_lamport<Pattern, T, kRanks, true, true>,
          kThreads, 0)
      : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &control_active, allreduce_fusion_kernel_twoshot_sync<Pattern, T, kRanks, true>,
          kThreads, 0);
  TORCH_CHECK(status == cudaSuccess);
  TORCH_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    input.get_device()) == cudaSuccess);
  // [M, stock_mode, candidate regs/stack/shared/active, control regs/stack/shared/active,
  //  SMs, dynamic workspace B].
  return {input.size(0), one ? 1 : 2, candidate.numRegs, int64_t(candidate.localSizeBytes),
          int64_t(candidate.sharedSizeBytes), candidate_active, control.numRegs,
          int64_t(control.localSizeBytes), int64_t(control.sharedSizeBytes), control_active,
          sm_count, workspace_bytes_for(input.size(0))};
}
}  // namespace mach_lossless_prefill_long

TORCH_LIBRARY(mach_lossless_prefill_long, m) {
  m.def("run(Tensor input, Tensor residual, Tensor gamma, Tensor workspace, Tensor(a!) residual_out, Tensor(b!) norm_out, int rank, int workspace_bytes, float eps, float weight_bias, bool pdl) -> ()");
  m.def("info(Tensor input) -> int[]");
}
TORCH_LIBRARY_IMPL(mach_lossless_prefill_long, CUDA, m) {
  m.impl("run", &mach_lossless_prefill_long::run);
  m.impl("info", &mach_lossless_prefill_long::info);
}
