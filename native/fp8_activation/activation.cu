// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Arithmetic adapted from vLLM v0.29.0 fused_silu_mul_block_quant.cu.
// One warp owns each group; four groups share a CTA without block barriers.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cmath>

namespace mach_fp8_activation {

template <bool Transposed>
__global__ void kernel(c10::Float8_e4m3fn* __restrict__ output,
                       const c10::BFloat16* __restrict__ input,
                       float* __restrict__ scales, int rows) {
  constexpr int kHidden = 9216;
  constexpr int kGroupSize = 128;
  constexpr int kGroups = kHidden / kGroupSize;
  const int lane = threadIdx.x & 31;
  const int group_id = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (group_id >= rows * kGroups) return;
  const int row = group_id / kGroups;
  const int group = group_id - row * kGroups;
  const int64_t base = int64_t(row) * kHidden * 2 + group * kGroupSize;
  float values[4];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int offset = lane + i * 32;
    float gate = static_cast<float>(input[base + offset]);
    float up = static_cast<float>(input[base + kHidden + offset]);
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    float silu_gate = gate * sigmoid_gate;
    float result = silu_gate * up;
    values[i] = result;
    maximum = fmaxf(maximum, fabsf(result));
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
  }
  float scale = fmaxf(maximum / 448.0f, 1.0f / (448.0f * 512.0f));
  if (lane == 0) {
    const int64_t offset = Transposed ? int64_t(group) * rows + row : group_id;
    scales[offset] = scale;
  }
  const int64_t out_base = int64_t(row) * kHidden + group * kGroupSize;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float value = values[i] / scale;
    float clamped = fmaxf(-448.0f, fminf(value, 448.0f));
    output[out_base + lane + i * 32] = c10::Float8_e4m3fn(clamped);
  }
}

void run(at::Tensor& out, const at::Tensor& input, at::Tensor& scales,
         bool is_scale_transposed) {
  TORCH_CHECK(input.is_cuda() && out.device() == input.device() &&
              scales.device() == input.device(), "Expected tensors on one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kFloat8_e4m3fn &&
              scales.scalar_type() == at::kFloat, "Expected BF16, E4M3, FP32 tensors");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2 && scales.dim() == 2);
  const int64_t rows = input.size(0);
  TORCH_CHECK(rows >= 1 && rows <= 2048 && input.size(1) == 18432);
  TORCH_CHECK(out.size(0) == rows && out.size(1) == 9216 &&
              scales.size(0) == rows && scales.size(1) == 72);
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(is_scale_transposed
                  ? scales.stride(0) == 1 && scales.stride(1) == rows
                  : scales.is_contiguous(), "Unexpected scale layout");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 12 && properties->minor == 0,
              "FP8 activation requires SM120");
  auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const int groups = int(rows) * 72;
  if (is_scale_transposed) {
    kernel<true><<<(groups + 3) / 4, 128, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(), input.data_ptr<c10::BFloat16>(),
        scales.data_ptr<float>(), int(rows));
  } else {
    kernel<false><<<(groups + 3) / 4, 128, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(), input.data_ptr<c10::BFloat16>(),
        scales.data_ptr<float>(), int(rows));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace mach_fp8_activation

TORCH_LIBRARY(mach_fp8_activation, m) {
  m.def("run(Tensor(a!) out, Tensor input, Tensor(b!) scales, "
        "bool is_scale_transposed) -> ()");
}
TORCH_LIBRARY_IMPL(mach_fp8_activation, CUDA, m) {
  m.impl("run", &mach_fp8_activation::run);
}
