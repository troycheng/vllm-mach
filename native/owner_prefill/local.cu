// Local TP2 ordered sum plus original residual/RMSNorm for one owner half.
// No communication, quantization, or GEMM is performed here.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include "trtllm_allreduce_fusion.cuh"

namespace mach_owner_local {
using namespace flashinfer::trtllm_allreduce_fusion;
using flashinfer::vec_t;
using T = __nv_bfloat16;
constexpr auto Pattern = AllReduceFusionPattern::kARResidualRMSNorm;
constexpr int kRows = 2048, kH = 5120, kRanks = 2, kVecSize = 8;
constexpr int kThreads = kH / kVecSize;
static_assert(kThreads == 640);

// `vals[0]` is TP rank 0 and `vals[1]` TP rank 1. allreduce_sum<...,true>
// retains the deployed FP32 add then BF16 conversion before FusedOp consumes it.
__global__ void ordered_sum_norm_kernel(AllReduceFusionParams<T> params,
                                        T const* part0, T const* part1) {
  IndexHelper<T> index_helper(params);
  int access_id = index_helper.access_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int access_stride = index_helper.access_stride;
  int token_id = index_helper.token_id;
  int token_stride = index_helper.token_stride;
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);
  cudaGridDependencySynchronize();
  for (int idx = access_id, token = token_id; idx < index_helper.tot_access;
       idx += access_stride, token += token_stride) {
    vec_t<T, kVecSize> vals[kRanks];
    vals[0].load(part0 + idx * kVecSize);
    vals[1].load(part1 + idx * kVecSize);
    vec_t<T, kVecSize> summed = allreduce_sum<T, 8, 2, true>(vals);
    fused_op.update(idx);
    fused_op(summed, token);
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

void ordered_sum_norm(at::Tensor const& part0, at::Tensor const& part1,
                      at::Tensor const& residual, at::Tensor const& gamma,
                      at::Tensor& residual_out, at::Tensor& norm_out,
                      double eps, double weight_bias, bool pdl) {
  for (auto const& tensor : {part0, part1, residual, gamma, residual_out, norm_out}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                tensor.scalar_type() == at::kBFloat16);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0);
    TORCH_CHECK(tensor.device() == part0.device());
  }
  TORCH_CHECK(part0.dim() == 2 && part0.size(0) == kRows && part0.size(1) == kH);
  TORCH_CHECK(part1.sizes() == part0.sizes() && residual.sizes() == part0.sizes() &&
              residual_out.sizes() == part0.sizes() && norm_out.sizes() == part0.sizes());
  TORCH_CHECK(gamma.dim() == 1 && gamma.numel() == kH);
  TORCH_CHECK(eps == 1e-6 && weight_bias == 1.0,
              "ordered_sum_norm fixes eps=1e-6 and weight_bias=1");

  c10::cuda::CUDAGuard guard(part0.device());
  int sm_count = 0, active = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                    part0.get_device()) == cudaSuccess && sm_count > 0,
              "ordered_sum_norm could not query a positive SM count");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active, ordered_sum_norm_kernel, kThreads, 0) == cudaSuccess && active >= 1,
              "ordered_sum_norm cannot sustain one 640-thread whole-token CTA per SM");

  AllReduceFusionParams<T> params{};
  params.nranks = kRanks;
  params.rank = 0;  // Local ordered inputs already represent TP ranks 0 then 1.
  params.size = int(part0.numel());
  params.hidden_dim = kH;
  params.allreduce_in = const_cast<void*>(part0.data_ptr());
  params.residual_in = const_cast<void*>(residual.data_ptr());
  params.residual_out = residual_out.data_ptr();
  params.norm_out = norm_out.data_ptr();
  params.rms_gamma = const_cast<void*>(gamma.data_ptr());
  params.rms_eps = float(eps);
  params.weight_bias = float(weight_bias);
  params.pattern = Pattern;
  params.stream = c10::cuda::getCurrentCUDAStream().stream();
  params.trigger_completion_at_end = true;

  cudaLaunchConfig_t config{};
  // One whole-token CTA per SM, with no idle CTAs past the input rows.
  config.gridDim = std::min(sm_count, int(part0.size(0)));
  config.blockDim = kThreads;
  config.stream = params.stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim = {1, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 2;
  TORCH_CHECK(cudaLaunchKernelEx(&config, ordered_sum_norm_kernel, params,
                                 reinterpret_cast<T const*>(part0.data_ptr()),
                                 reinterpret_cast<T const*>(part1.data_ptr())) == cudaSuccess,
              "ordered_sum_norm launch failed");
}
}  // namespace mach_owner_local

TORCH_LIBRARY(mach_owner_local, m) {
  m.def("ordered_sum_norm(Tensor part0, Tensor part1, Tensor residual, Tensor gamma, "
        "Tensor(a!) residual_out, Tensor(b!) norm_out, float eps, float weight_bias, bool pdl) -> ()");
}
TORCH_LIBRARY_IMPL(mach_owner_local, CUDA, m) {
  m.impl("ordered_sum_norm", &mach_owner_local::ordered_sum_norm);
}
