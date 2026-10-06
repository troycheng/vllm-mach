// SPDX-License-Identifier: Apache-2.0
// The original file and all its helpers are included from the exact checkout.
// Rename symbol identifiers to prevent ELF interposition against working vLLM.
// Kernel arithmetic and the original stable host wrapper remain unchanged.
#define vllm pr45055_exact_impl
#define silu_and_mul_per_block_quant pr45055_silu_and_mul_per_block_quant
#include "libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
#undef silu_and_mul_per_block_quant
#undef vllm
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(pr45055_exact, m) {
  m.def("run(Tensor(a!) out, Tensor input, Tensor(b!) scales, int group_size, "
        "Tensor? scale_ub, bool is_scale_transposed) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(pr45055_exact, CUDA, m) {
  m.impl("run", TORCH_BOX(&pr45055_silu_and_mul_per_block_quant));
}
