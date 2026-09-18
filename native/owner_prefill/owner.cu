// TP2 owner-only residual/RMSNorm and MXFP8-byte gather probe.
// Derived from c32_pack_refine_20260907/direct_kernel/ar_codec_extension.cu.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include "trtllm_allreduce_fusion.cuh"

namespace mach_owner {
using namespace flashinfer::trtllm_allreduce_fusion;
using flashinfer::vec_t;
using T = __nv_bfloat16;
constexpr auto Pattern = AllReduceFusionPattern::kARResidualRMSNorm;
#include "codec_helpers.cuh"
#include "sum_codec_helpers.cuh"

constexpr int kM = 4096, kH = 5120, kRanks = 2, kVecSize = 8;
constexpr int kInputElements = kM * kH;
constexpr int kInputPayloadBytes = kInputElements * int(sizeof(T));
constexpr int kRequiredWorkspaceBytes =
    kInputElements * int(sizeof(T)) * 2 +
    kInputElements / 256 * int(sizeof(unsigned short)) * 2;
static_assert(kInputPayloadBytes == 40 * 1024 * 1024);
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

// Reference peer-half pack, barrier 1, own-half FP32 sum/BF16 and FusedOp.
// Barrier 2 protects workspace from reuse before the peer finishes decode.
__global__ void reduce_owner_kernel(AllReduceFusionParams<T> params) {
  constexpr int VEC_SIZE = kVecSize, NRanks = kRanks;
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int own_access_base = params.rank * params.size / NRanks / VEC_SIZE;
  FusedOp<Pattern, T> fused_op(params, access_id + own_access_base,
                               access_id_in_token);
  cudaGridDependencySynchronize();
  SyncComm<NRanks> comm(params.workspace);
  int half_tokens = params.size / params.hidden_dim / NRanks;
  int header_offset = params.size * sizeof(T) * 2;

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
  for (int idx = own_access_id, tidx = token_id + own_begin;
       idx < own_tot_access; idx += access_stride, tidx += token_stride) {
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      if (r == params.rank) {
        vals[r].load(reinterpret_cast<T const*>(params.allreduce_in) + idx * VEC_SIZE);
      } else {
        vals[r] = decode_peer_input(comm.comm_bufs[r], idx, header_offset);
      }
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, true>(vals);
    fused_op.update(idx);
    fused_op(sum_val, tidx);
  }
  barrier.sync();
  comm.update(barrier.m_flag_value);
  cudaTriggerProgrammaticLaunchCompletion();
}

// Opaque two-payload gather. Workspace holds rank0 values/scales then rank1.
__global__ void gather_mx8_kernel(void** workspace, uint8_t const* local_values,
                                  uint8_t const* local_scales, uint8_t* out_values,
                                  uint8_t* out_scales, int rank, int values_bytes,
                                  int scales_bytes) {
  constexpr int NRanks = kRanks;
  int values_vec = values_bytes / int(sizeof(uint4));
  int scales_vec = scales_bytes / int(sizeof(uint4));
  int packet_vec = values_vec + scales_vec;
  int peer = 1 - rank;
  cudaGridDependencySynchronize();
  SyncComm<NRanks> comm(workspace);
  auto* remote_packet = reinterpret_cast<uint4*>(comm.comm_bufs[peer]) +
                        rank * packet_vec;
  auto* out_values4 = reinterpret_cast<uint4*>(out_values);
  auto* out_scales4 = reinterpret_cast<uint4*>(out_scales);
  auto const* local_values4 = reinterpret_cast<uint4 const*>(local_values);
  auto const* local_scales4 = reinterpret_cast<uint4 const*>(local_scales);
  for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < packet_vec;
       i += int(blockDim.x * gridDim.x)) {
    if (i < values_vec) {
      uint4 x = local_values4[i];
      out_values4[rank * values_vec + i] = x;
      remote_packet[i] = x;
    } else {
      int j = i - values_vec;
      uint4 x = local_scales4[j];
      out_scales4[rank * scales_vec + j] = x;
      remote_packet[i] = x;
    }
  }
  Barrier<NRanks> barrier(rank, comm);
  barrier.sync();
  auto const* local_packet = reinterpret_cast<uint4 const*>(comm.comm_bufs[rank]) +
                             peer * packet_vec;
  for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < packet_vec;
       i += int(blockDim.x * gridDim.x)) {
    if (i < values_vec) {
      out_values4[peer * values_vec + i] = local_packet[i];
    } else {
      out_scales4[peer * scales_vec + i - values_vec] = local_packet[i];
    }
  }
  barrier.sync();
  comm.update(barrier.m_flag_value);
  cudaTriggerProgrammaticLaunchCompletion();
}

void reduce_owner(at::Tensor const& input, at::Tensor const& residual,
                  at::Tensor const& gamma, at::Tensor const& workspace,
                  at::Tensor& residual_out, at::Tensor& norm_out, int64_t rank,
                  int64_t workspace_bytes, double eps, double weight_bias,
                  bool pdl) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(0) == kM &&
              input.size(1) == kH);
  for (auto const& x : {input, residual, gamma, residual_out, norm_out}) {
    TORCH_CHECK(x.device() == input.device() && x.is_contiguous() &&
                x.scalar_type() == at::kBFloat16);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(residual.sizes() == input.sizes() &&
              residual_out.sizes() == input.sizes() && norm_out.sizes() == input.sizes());
  TORCH_CHECK(gamma.numel() == kH && rank >= 0 && rank < kRanks);
  TORCH_CHECK(eps == 1e-6 && weight_bias == 1.0,
              "probe fixes eps=1e-6 and weight_bias=1");
  TORCH_CHECK(workspace.device() == input.device() && workspace.is_contiguous() &&
              workspace.scalar_type() == at::kLong && workspace.numel() >= 7);
  TORCH_CHECK(workspace_bytes >= kRequiredWorkspaceBytes);
  c10::cuda::CUDAGuard guard(input.device());
  AllReduceFusionParams<T> params{};
  params.nranks = kRanks; params.rank = int(rank); params.size = int(input.numel());
  params.hidden_dim = kH; params.workspace = reinterpret_cast<void**>(workspace.data_ptr());
  params.allreduce_in = input.data_ptr(); params.residual_in = residual.data_ptr();
  params.residual_out = residual_out.data_ptr(); params.norm_out = norm_out.data_ptr();
  params.rms_gamma = gamma.data_ptr(); params.rms_eps = float(eps);
  params.weight_bias = float(weight_bias); params.use_oneshot = false;
  params.stream = c10::cuda::getCurrentCUDAStream().stream();
  params.pattern = Pattern; params.trigger_completion_at_end = true;
  int active = 0, sm_count = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, reduce_owner_kernel, 640, 0) == cudaSuccess && active >= 1,
      "owner reduce cannot sustain the original 640-thread whole-token block");
  TORCH_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    input.get_device()) == cudaSuccess && sm_count > 0,
              "owner reduce could not query a positive SM count");
  cudaLaunchConfig_t config{};
  // All CTAs must be resident and have a slot in the barrier workspace.
  config.gridDim = std::min(sm_count, details::kBarrierFlagCount);
  config.blockDim = 640; config.stream = params.stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim = {1, 1, 1};
  config.attrs = attrs; config.numAttrs = 2;
  TORCH_CHECK(cudaLaunchKernelEx(&config, reduce_owner_kernel, params) == cudaSuccess,
              "reduce_owner launch failed");
}

void gather_mx8(at::Tensor const& local_values, at::Tensor const& local_scales,
                at::Tensor const& workspace, at::Tensor& out_values,
                at::Tensor& out_scales, int64_t rank, int64_t workspace_bytes,
                bool pdl) {
  for (auto const& x : {local_values, local_scales, workspace, out_values, out_scales}) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
  }
  TORCH_CHECK(local_values.scalar_type() == at::kByte &&
              local_scales.scalar_type() == at::kByte &&
              out_values.scalar_type() == at::kByte && out_scales.scalar_type() == at::kByte);
  TORCH_CHECK(workspace.scalar_type() == at::kLong && workspace.numel() >= 7);
  TORCH_CHECK(local_values.device() == local_scales.device() &&
              workspace.device() == local_values.device() &&
              out_values.device() == local_values.device() &&
              out_scales.device() == local_values.device());
  TORCH_CHECK(rank >= 0 && rank < kRanks && workspace_bytes >= kRequiredWorkspaceBytes);
  int64_t values_bytes = local_values.numel(), scales_bytes = local_scales.numel();
  TORCH_CHECK(values_bytes > 0 && scales_bytes > 0 && values_bytes % 16 == 0 &&
              scales_bytes % 16 == 0);
  TORCH_CHECK(out_values.numel() == 2 * values_bytes &&
              out_scales.numel() == 2 * scales_bytes);
  TORCH_CHECK(2 * (values_bytes + scales_bytes) <= kInputPayloadBytes);
  for (auto const& x : {local_values, local_scales, out_values, out_scales}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(local_values.data_ptr() != out_values.data_ptr() &&
              local_scales.data_ptr() != out_scales.data_ptr(),
              "gather output may not alias local packet");
  c10::cuda::CUDAGuard guard(local_values.device());
  int sm_count = 0, active = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    local_values.get_device()) == cudaSuccess &&
              sm_count > 0,
              "gather_mx8 could not query a positive SM count");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, gather_mx8_kernel, 256, 0) == cudaSuccess && active >= 1,
      "gather_mx8 cannot sustain one 256-thread block per SM");
  cudaLaunchConfig_t config{};
  // All CTAs must be resident and have a slot in the barrier workspace.
  config.gridDim = std::min(sm_count, details::kBarrierFlagCount);
  config.blockDim = 256;
  config.stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim = {1, 1, 1};
  config.attrs = attrs; config.numAttrs = 2;
  TORCH_CHECK(cudaLaunchKernelEx(
                  &config, gather_mx8_kernel,
                  reinterpret_cast<void**>(workspace.data_ptr()),
                  reinterpret_cast<uint8_t const*>(local_values.data_ptr()),
                  reinterpret_cast<uint8_t const*>(local_scales.data_ptr()),
                  reinterpret_cast<uint8_t*>(out_values.data_ptr()),
                  reinterpret_cast<uint8_t*>(out_scales.data_ptr()), int(rank),
                  int(values_bytes), int(scales_bytes)) == cudaSuccess,
              "gather_mx8 launch failed");
}
}  // namespace mach_owner

TORCH_LIBRARY(mach_owner, m) {
  m.def("reduce_owner(Tensor input, Tensor residual, Tensor gamma, Tensor workspace, Tensor(a!) residual_out, Tensor(b!) norm_out, int rank, int workspace_bytes, float eps, float weight_bias, bool pdl) -> ()");
  m.def("gather_mx8(Tensor local_values, Tensor local_scales, Tensor workspace, Tensor(a!) out_values, Tensor(b!) out_scales, int rank, int workspace_bytes, bool pdl) -> ()");
}
TORCH_LIBRARY_IMPL(mach_owner, CUDA, m) {
  m.impl("reduce_owner", &mach_owner::reduce_owner);
  m.impl("gather_mx8", &mach_owner::gather_mx8);
}
