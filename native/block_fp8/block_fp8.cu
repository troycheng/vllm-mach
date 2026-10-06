// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
namespace mach_block_fp8 {
template<class T> using enable_sm120_family = T;
namespace c3x {
template <typename Kernel>
void cutlass_gemm_caller(at::Device device, cute::Shape<int,int,int,int> problem,
    typename Kernel::MainloopArguments mainloop,
    typename Kernel::EpilogueArguments epilogue,
    typename Kernel::TileSchedulerArguments scheduler) {
  const c10::cuda::CUDAGuard guard(device);
  cutlass::KernelHardwareInfo hw;
  typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
      problem, mainloop, epilogue, hw, scheduler};
  cutlass::gemm::device::GemmUniversalAdapter<Kernel> op;
  auto status = op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
  auto workspace = at::empty({int64_t(op.get_workspace_size(args))}, at::TensorOptions().dtype(at::kByte).device(device));
  status = op.run(args, workspace.data_ptr(), c10::cuda::getCurrentCUDAStream(device.index()).stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}
}
}
namespace mach_block_fp8 {

using namespace cute;

// clang-format off
template <class OutType, int ScaleGranularityM,
          int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape,
          class EpilogueScheduler, class MainloopScheduler,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_fp8_blockwise {
  static constexpr bool swap_ab = swap_ab_;
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  // ColumnMajor is used for B to match the CUTLASS convention.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void; // TODO: support bias
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = conditional_t<swap_ab,
      cutlass::detail::Sm120BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        cute::UMMA::Major::K, cute::UMMA::Major::MN>,
      cutlass::detail::Sm120BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        cute::UMMA::Major::MN, cute::UMMA::Major::K>>;

  // layout_SFA and layout_SFB cannot be swapped since they are deduced.
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using ElementScalar = float;
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      conditional_t<swap_ab, LayoutC_Transpose, LayoutC>,
      AlignmentC,
      ElementD,
      conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
      AlignmentD,
      EpilogueScheduler,
      DefaultOperation
  >::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using CollectiveMainloop = conditional_t<swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          cute::tuple<LayoutB_Transpose, LayoutSFA>,
          AlignmentB,
          ElementA,
          cute::tuple<LayoutA_Transpose, LayoutSFB>,
          AlignmentA,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler
      >::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          cute::tuple<LayoutA, LayoutSFA>,
          AlignmentA,
          ElementB,
          cute::tuple<LayoutB, LayoutSFB>,
          AlignmentB,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler
      >::CollectiveOp>;

  // SM12x family to support both SM120 (RTX 5090) and SM121 (DGX Spark)
  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(at::Tensor& out, at::Tensor const& a,
                                   at::Tensor const& b,
                                   at::Tensor const& a_scales,
                                   at::Tensor const& b_scales, int swizzle) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;
  using ScaleConfig = typename Gemm::ScaleConfig;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = typename Gemm::ElementBlockScale;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);

  StrideA a_stride;
  StrideB b_stride;
  StrideC c_stride;
  a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  LayoutSFA layout_SFA = swap_ab ?
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1)) :
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = swap_ab ?
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1)) :
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto a_scales_ptr = static_cast<ElementBlockScale const*>(a_scales.data_ptr());
  auto b_scales_ptr = static_cast<ElementBlockScale const*>(b_scales.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.layout_SFA = layout_SFA;
  mainloop_args.layout_SFB = layout_SFB;
  if (swap_ab) {
    mainloop_args.ptr_A = b_ptr;
    mainloop_args.dA = b_stride;
    mainloop_args.ptr_B = a_ptr;
    mainloop_args.dB = a_stride;
    mainloop_args.ptr_SFA = b_scales_ptr;
    mainloop_args.ptr_SFB = a_scales_ptr;
  } else {
    mainloop_args.ptr_A = a_ptr;
    mainloop_args.dA = a_stride;
    mainloop_args.ptr_B = b_ptr;
    mainloop_args.dB = b_stride;
    mainloop_args.ptr_SFA = a_scales_ptr;
    mainloop_args.ptr_SFB = b_scales_ptr;
  }
  auto prob_shape = swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};
  typename GemmKernel::TileSchedulerArguments scheduler{};
  scheduler.max_swizzle_size = swizzle;
  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args, epilogue_args, scheduler);
}


using BF = cutlass::bfloat16_t;
using NarrowPing = cutlass_3x_gemm_fp8_blockwise<BF,64,1,128,Shape<_64,_32,_128>,Shape<_1,_1,_1>,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120,true>;

at::Tensor mm(const at::Tensor& a, const at::Tensor& b,
              const at::Tensor& sa, const at::Tensor& sb) {
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.is_contiguous());
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn &&
              b.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(b.dim() == 2 && b.stride(0) == 1 &&
              b.stride(1) == b.size(0));
  TORCH_CHECK(a.size(0) >= 16 && a.size(0) <= 128 && a.size(0) % 8 == 0);
  TORCH_CHECK(b.size(1) == 2560 && a.size(1) == b.size(0) &&
              (a.size(1) == 4096 || a.size(1) == 9216));
  TORCH_CHECK(sa.dim() == 2 && sb.dim() == 2 &&
              sa.scalar_type() == at::kFloat && sb.scalar_type() == at::kFloat);
  TORCH_CHECK(sa.device() == a.device() && sb.device() == a.device() &&
              b.device() == a.device());
  TORCH_CHECK(sa.size(0) == a.size(0) && sa.size(1) == a.size(1) / 128 &&
              sa.stride(0) == 1 && sa.stride(1) == a.size(0));
  TORCH_CHECK(sb.size(0) == a.size(1) / 128 && sb.size(1) == 40 &&
              sb.stride(0) == 1 && sb.stride(1) == sb.size(0));
  auto out = at::empty({a.size(0), b.size(1)},
                       a.options().dtype(at::kBFloat16));
  cutlass_gemm_caller_blockwise<NarrowPing>(out, a, b, sa, sb, 1);
  return out;
}
}
TORCH_LIBRARY(vllm_mach_block_fp8, m) {
  m.def("mm(Tensor a, Tensor b, Tensor sa, Tensor sb) -> Tensor");
}
TORCH_LIBRARY_IMPL(vllm_mach_block_fp8, CUDA, m) {
  m.impl("mm", &mach_block_fp8::mm);
}
