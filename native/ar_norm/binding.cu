#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <string>
#include "trtllm_allreduce_fusion_mxfp8.cuh"

namespace ar = flashinfer::trtllm_allreduce_fusion;

void run(const at::Tensor& hidden, const at::Tensor& residual,
         const at::Tensor& gamma, const at::Tensor& workspace,
         at::Tensor norm, at::Tensor residual_out, at::Tensor quant,
         at::Tensor scales, int64_t rank, double eps, bool fused) {
  TORCH_CHECK(hidden.is_cuda() && hidden.dim() == 2 && hidden.size(1) == 5120,
              "mach_norm_quant.run requires a CUDA [M, 5120] hidden tensor");
  const auto* properties = at::cuda::getDeviceProperties(hidden.get_device());
  TORCH_CHECK(properties->major == 12 && properties->minor == 0,
              "mach_norm_quant.run requires an SM120 GPU");
  TORCH_CHECK(hidden.size(0) > 0 && hidden.size(0) <= 32 && rank >= 0 && rank < 2,
              "mach_norm_quant.run requires 1 <= M <= 32 and TP2 rank in [0, 1]");
  for (const auto& tensor : {hidden, residual, gamma, norm, residual_out}) {
    TORCH_CHECK(tensor.device() == hidden.device() && tensor.is_contiguous(),
                "mach_norm_quant.run BF16 tensors must be contiguous and on hidden's device");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16,
                "mach_norm_quant.run requires BF16 hidden, residual, gamma, norm, and residual_out");
  }
  TORCH_CHECK(residual.sizes() == hidden.sizes() && norm.sizes() == hidden.sizes()
              && residual_out.sizes() == hidden.sizes() && gamma.numel() == 5120,
              "mach_norm_quant.run requires residual, norm, and residual_out to match hidden and gamma.numel() == 5120");
  TORCH_CHECK(workspace.device() == hidden.device() && workspace.is_contiguous()
              && workspace.scalar_type() == at::kLong && workspace.numel() == 7,
              "mach_norm_quant.run requires a contiguous int64 workspace with 7 entries");
  TORCH_CHECK(quant.device() == hidden.device() && quant.scalar_type() == at::kByte
              && quant.is_contiguous() && quant.sizes() == hidden.sizes(),
              "mach_norm_quant.run requires a contiguous uint8 quant tensor matching hidden");
  TORCH_CHECK(scales.device() == hidden.device() && scales.scalar_type() == at::kByte
              && scales.is_contiguous() && scales.numel() == 128 * 160,
              "mach_norm_quant.run requires a contiguous uint8 scale tensor with 20480 entries");
  c10::cuda::CUDAGuard guard(hidden.device());
  ar::AllReduceFusionParams<__nv_bfloat16> p{};
  p.nranks = 2;
  p.rank = static_cast<int>(rank);
  p.size = static_cast<int>(hidden.numel());
  p.hidden_dim = 5120;
  p.workspace = reinterpret_cast<void**>(workspace.data_ptr());
  p.allreduce_in = hidden.data_ptr();
  p.residual_in = residual.data_ptr();
  p.residual_out = residual_out.data_ptr();
  p.norm_out = norm.data_ptr();
  p.quant_out = quant.data_ptr();
  p.scale_out = scales.data_ptr();
  p.rms_gamma = gamma.data_ptr();
  p.rms_eps = static_cast<float>(eps);
  p.weight_bias = 1.0f;
  p.use_oneshot = true;
  p.stream = at::cuda::getCurrentCUDAStream();
  p.trigger_completion_at_end = true;
  if (fused) {
    p.pattern = ar::AllReduceFusionPattern::kARResidualRMSNormOutMXFP8;
    C10_CUDA_CHECK((ar::allreduce_fusion_kernel_launcher<
        ar::AllReduceFusionPattern::kARResidualRMSNormOutMXFP8,
        __nv_bfloat16, 2, true>(p, true)));
  } else {
    p.pattern = ar::AllReduceFusionPattern::kARResidualRMSNorm;
    C10_CUDA_CHECK((ar::allreduce_fusion_kernel_launcher<
        ar::AllReduceFusionPattern::kARResidualRMSNorm,
        __nv_bfloat16, 2, true>(p, true)));
  }
}

TORCH_LIBRARY(mach_norm_quant, m) {
  m.def("run(Tensor hidden, Tensor residual, Tensor gamma, Tensor workspace, "
        "Tensor(a!) norm, Tensor(b!) residual_out, Tensor(c!) quant, Tensor(d!) scales, "
        "int rank, float eps, bool fused) -> ()");
  m.def("abi() -> str", []() { return std::string("ar-norm-mxfp8-v1"); });
}
TORCH_LIBRARY_IMPL(mach_norm_quant, CUDA, m) {
  m.impl("run", &run);
}
