// SPDX-License-Identifier: Apache-2.0
// Complete TP2 BF16 AllReduce + residual + Gemma RMSNorm boundary prototype.
// Adds lossless compression of the sum peer-write leg to the input codec.
// Original FP32 accumulation, BF16 conversion, barriers and FusedOp remain.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <vector>
#include "trtllm_allreduce_fusion.cuh"

namespace mach_lossless_prefill_sum {
using namespace flashinfer::trtllm_allreduce_fusion;
using flashinfer::vec_t;
using T = __nv_bfloat16;
constexpr auto Pattern = AllReduceFusionPattern::kARResidualRMSNorm;
#include "codec_helpers.cuh"
#include "sum_codec_helpers.cuh"

__device__ __forceinline__ vec_t<T, 8> decode_peer(void const* buffer, int idx,
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

__global__ void packed_twoshot(AllReduceFusionParams<T> params) {
  constexpr int VEC_SIZE = 8, NRanks = 2;
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);
  cudaGridDependencySynchronize();
  SyncComm<NRanks> comm(params.workspace);
  // The preceding pack kernel fully publishes this rank's input and header.
  // The original cross-rank release/acquire barriers remain unchanged.
  Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();
  int half_tokens = params.size / params.hidden_dim / NRanks;
  int begin_token = params.rank * half_tokens;
  int comm_access_id = access_id + begin_token * params.hidden_dim / VEC_SIZE;
  int comm_tot_access = (begin_token + half_tokens) * params.hidden_dim / VEC_SIZE;
  int header_offset = params.size * sizeof(T) * 2;
  int sum_header_relative = params.size * sizeof(T) + params.size / 256 * sizeof(unsigned short);
  for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> vals[NRanks];
    #pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      if (r == params.rank)
        vals[r].load(reinterpret_cast<T const*>(params.allreduce_in) + idx * VEC_SIZE);
      else
        vals[r] = decode_peer(comm.comm_bufs[r], idx, header_offset);
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, true>(vals);
    // Keep this rank's produced half raw for its local consumer. Only the copy
    // sent to the peer is packed, in fixed raw-sized slots in the sum region.
    sum_val.store(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                  (tot_access + idx) * VEC_SIZE);
    store_packed_sum(sum_val,
        reinterpret_cast<T*>(comm.comm_bufs[1 - params.rank]) + tot_access * VEC_SIZE,
        idx, sum_header_relative);
  }
  barrier.sync();
  #pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int start = access_id + r * half_tokens * params.hidden_dim / VEC_SIZE;
    int tidx_start = token_id + r * half_tokens;
    int end = (r + 1) * half_tokens * params.hidden_dim / VEC_SIZE;
    for (int idx = start, tidx = tidx_start; idx < end;
         idx += access_stride, tidx += token_stride) {
      fused_op.update(idx);
      vec_t<T, VEC_SIZE> sum_val;
      if (r == params.rank) {
        sum_val.load(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                     (tot_access + idx) * VEC_SIZE);
      } else {
        // The original second barrier publishes both remote payload and header.
        sum_val = decode_peer(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                              tot_access * VEC_SIZE, idx, sum_header_relative);
      }
      fused_op(sum_val, tidx);
    }
  }
  comm.update(barrier.m_flag_value);
  cudaTriggerProgrammaticLaunchCompletion();
}

void run(at::Tensor const& input, at::Tensor const& residual,
         at::Tensor const& gamma, at::Tensor const& workspace,
         at::Tensor const& residual_out, at::Tensor const& norm_out,
         int64_t rank, int64_t workspace_bytes, double eps, double weight_bias,
         bool packed, bool pdl) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(0) == 4096 && input.size(1) == 5120);
  for (auto const& x : {input, residual, gamma, residual_out, norm_out}) {
    TORCH_CHECK(x.device() == input.device() && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
  }
  TORCH_CHECK(residual.sizes() == input.sizes() && residual_out.sizes() == input.sizes() && norm_out.sizes() == input.sizes());
  TORCH_CHECK(gamma.numel() == 5120 && rank >= 0 && rank < 2 && eps > 0);
  TORCH_CHECK(workspace.device() == input.device() && workspace.is_contiguous() && workspace.scalar_type() == at::kLong && workspace.numel() >= 7);
  // Input and sum headers follow BOTH raw-sized payload regions, without overlap.
  TORCH_CHECK(workspace_bytes >= input.numel() * 4 + input.numel() / 256 * 4);
  c10::cuda::CUDAGuard guard(input.device());
  AllReduceFusionParams<T> params{};
  params.nranks = 2; params.rank = int(rank);
  params.size = int(input.numel()); params.hidden_dim = 5120;
  params.workspace = reinterpret_cast<void**>(workspace.data_ptr());
  params.allreduce_in = input.data_ptr(); params.residual_in = residual.data_ptr();
  params.residual_out = residual_out.data_ptr(); params.norm_out = norm_out.data_ptr();
  params.rms_gamma = gamma.data_ptr(); params.rms_eps = float(eps);
  params.weight_bias = float(weight_bias); params.use_oneshot = false;
  params.stream = c10::cuda::getCurrentCUDAStream().stream();
  params.pattern = Pattern; params.trigger_completion_at_end = true;
  if (!packed) {
    auto error = allreduce_fusion_kernel_launcher<Pattern, T, 2, true>(params, pdl);
    TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
    return;
  }
  int active = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, packed_twoshot, 640, 0) == cudaSuccess && active >= 1,
              "packed kernel cannot sustain the original640-thread whole-token block");
  int groups = int(input.numel() / 256);
  pack_workspace<<<(groups + 7) / 8, 256, 0, params.stream>>>(
      reinterpret_cast<uint4 const*>(input.data_ptr()), params.workspace, int(rank),
      int(input.numel() * 4), groups);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "workspace pack launch failed");
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
  auto error = cudaLaunchKernelEx(&config, packed_twoshot, params);
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

std::vector<int64_t> info(at::Tensor const& input) {
  c10::cuda::CUDAGuard guard(input.device());
  cudaFuncAttributes packed{}, control{};
  TORCH_CHECK(cudaFuncGetAttributes(&packed, packed_twoshot) == cudaSuccess);
  TORCH_CHECK(cudaFuncGetAttributes(&control, allreduce_fusion_kernel_twoshot_sync<Pattern, T, 2, true>) == cudaSuccess);
  int active = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, packed_twoshot, 640, 0) == cudaSuccess);
  return {packed.numRegs, int64_t(packed.localSizeBytes), int64_t(packed.sharedSizeBytes),
          control.numRegs, int64_t(control.localSizeBytes), int64_t(control.sharedSizeBytes), active};
}
}  // namespace mach_lossless_prefill_sum

TORCH_LIBRARY(mach_lossless_prefill_sum, m) {
  m.def("run(Tensor input, Tensor residual, Tensor gamma, Tensor workspace, Tensor(a!) residual_out, Tensor(b!) norm_out, int rank, int workspace_bytes, float eps, float weight_bias, bool packed, bool pdl) -> ()");
  m.def("info(Tensor input) -> int[]");
}
TORCH_LIBRARY_IMPL(mach_lossless_prefill_sum, CUDA, m) {
  m.impl("run", &mach_lossless_prefill_sum::run);
  m.impl("info", &mach_lossless_prefill_sum::info);
}
