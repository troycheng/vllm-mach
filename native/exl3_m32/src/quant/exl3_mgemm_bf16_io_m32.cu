#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

#include "util.h"
#include "util.cuh"
#include "quant/exl3_mgemm_bf16_io.cuh"
#include "quant/exl3_kernel_map.cuh"
#include "quant/hadamard_inner.cuh"
#include "exl3_gemm_inner_m32.cuh"

static void check_cuda_tensor_m32
(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name
)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on A.device()");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ __launch_bounds__(32)
static void exl3_bf16_grouped_had_m32_transform
(
    const __nv_bfloat16* __restrict__ A,
    half* __restrict__ A_had,
    const half** __restrict__ suh_list,
    const int size_m,
    const int size_k
)
{
    int group = blockIdx.z;
    int row = blockIdx.x;
    int col = blockIdx.y * 128;
    had_bh_r_128_inner<true, false>
    (
        A + row * size_k + col,
        A_had + ((size_t) group * size_m + row) * size_k + col,
        suh_list[group],
        0.088388347648f
    );
}

__global__ __launch_bounds__(32)
static void exl3_bf16_single_had_m32_transform
(
    const __nv_bfloat16* __restrict__ A,
    half* __restrict__ A_had,
    const half* __restrict__ suh,
    const int size_k
)
{
    int row = blockIdx.x;
    int col = blockIdx.y * 128;
    had_bh_r_128_inner<true, false>
    (
        A + row * size_k + col,
        A_had + row * size_k + col,
        suh,
        0.088388347648f
    );
}

template<int bits, int cb>
__global__ __launch_bounds__(512)
void exl3_mgemm_bf16_io_grouped_had_m32_kernel
(
    const uint16_t** __restrict__ B_list,
    half* __restrict__ C_scratch,
    __nv_bfloat16* __restrict__ C_bf16,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ A_had,
    const half** __restrict__ svh_list,
    const int* __restrict__ had_group_ids,
    const int count,
    const int group_count,
    const int output_stride
)
{
    int j = blockIdx.z;
    const uint16_t* B = B_list[j];
    int had_group_id = had_group_ids[j];
    if (had_group_id < 0 || had_group_id >= group_count) asm("trap;");
    const half* A_had_j = A_had + (size_t) had_group_id * size_m * size_k;
    half* C_j = C_scratch + (size_t) j * size_m * size_n;
    int lock_offs = blockIdx.z * size_n / 128;
    const half* svh = svh_list[j];
    __nv_bfloat16* C_bf16_j = C_bf16 + j * size_n;
    exl3_gemm_kernel_inner_m32
    <bits, false, cb, 32, 32, 128, 4, 3, true, true>
    (A_had_j, B, C_j, size_m, size_k, size_n,
     locks + lock_offs, svh, C_bf16_j, output_stride);
}

template<int bits, int cb>
__global__ __launch_bounds__(512)
void exl3_gemm_bf16_io_single_had_m32_fp16_kernel
(
    const uint16_t* __restrict__ B,
    half* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ A_had,
    const half* __restrict__ svh
)
{
    exl3_gemm_kernel_inner_m32
    <bits, false, cb, 32, 32, 128, 4, 3, true, true>
    (A_had, B, C, size_m, size_k, size_n, locks, svh);
}

template<int bits, int cb>
static void launch_m32
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& unique_suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& had_group_ids,
    const at::Tensor& locks,
    int force_num_sms,
    int output_stride
)
{
    constexpr int block_dim = 512;
    constexpr int smem_bytes = 90 * 1024;
    void* kernel = reinterpret_cast<void*>(
        exl3_mgemm_bf16_io_grouped_had_m32_kernel<bits, cb>);
    int device = A.get_device();
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, device));
    int total_sms = props.multiProcessorCount;
    int count = B.numel();
    int group_count = unique_suh.numel();
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = C_scratch.size(2);
    TORCH_CHECK(size_m == 32, "M32 BF16 MGEMM requires exactly 32 rows");
    TORCH_CHECK(force_num_sms > 0 && force_num_sms * count <= total_sms,
                "M32 grouped-Had BF16 MGEMM requires all matrices resident");
    int* locks_ptr = reinterpret_cast<int*>(locks.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cuda_check(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    const uint16_t** B_ptrs = reinterpret_cast<const uint16_t**>(B.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(C_scratch.data_ptr());
    __nv_bfloat16* C_bf16_ptr = reinterpret_cast<__nv_bfloat16*>(C_bf16.data_ptr());
    half* A_had_ptr = reinterpret_cast<half*>(A_had.data_ptr());
    const half** svh_ptrs = reinterpret_cast<const half**>(svh.data_ptr());
    const int* ids = reinterpret_cast<const int*>(had_group_ids.data_ptr());
    const half** unique_suh_ptrs = reinterpret_cast<const half**>(unique_suh.data_ptr());
    const __nv_bfloat16* A_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    const size_t locks_bytes =
        static_cast<size_t>(count) * (size_n / 128) * sizeof(int);
    cuda_check(cudaMemsetAsync(locks_ptr, 0, locks_bytes, stream));
    exl3_bf16_grouped_had_m32_transform<<<dim3(size_m, size_k / 128, group_count), 32, 0, stream>>>
    (A_ptr, A_had_ptr, unique_suh_ptrs, size_m, size_k);
    cuda_check(cudaPeekAtLastError());
    void* args[] = {
        &B_ptrs, &C_ptr, &C_bf16_ptr, &size_m, &size_k, &size_n, &locks_ptr,
        &A_had_ptr, &svh_ptrs, &ids, &count, &group_count, &output_stride,
    };
    cuda_check(cudaLaunchCooperativeKernel(
        kernel, dim3(force_num_sms, 1, count), dim3(block_dim), args,
        smem_bytes, stream));
    cuda_check(cudaPeekAtLastError());
}

template<int bits, int cb>
static void launch_single_m32_fp16
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& locks,
    int force_num_sms
)
{
    constexpr int block_dim = 512;
    constexpr int smem_bytes = 90 * 1024;
    void* kernel = reinterpret_cast<void*>(
        exl3_gemm_bf16_io_single_had_m32_fp16_kernel<bits, cb>);
    int device = A.get_device();
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, device));
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = C.size(1);
    int num_sms = force_num_sms ? force_num_sms : props.multiProcessorCount;
    TORCH_CHECK(num_sms > 0 && num_sms <= props.multiProcessorCount,
                "M32 single-matrix BF16 MGEMM requires a resident grid");
    TORCH_CHECK(num_sms <= (size_k / 32) * (size_n / 128),
                "force_num_sms exceeds the available KxN tiles");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16* A_ptr =
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(B.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(C.data_ptr());
    const half* suh_ptr = reinterpret_cast<const half*>(suh.data_ptr());
    half* A_had_ptr = reinterpret_cast<half*>(A_had.data_ptr());
    const half* svh_ptr = reinterpret_cast<const half*>(svh.data_ptr());
    int* locks_ptr = reinterpret_cast<int*>(locks.data_ptr());
    const size_t locks_bytes = static_cast<size_t>(size_n / 128) * sizeof(int);

    cuda_check(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    cuda_check(cudaMemsetAsync(locks_ptr, 0, locks_bytes, stream));
    exl3_bf16_single_had_m32_transform<<<dim3(size_m, size_k / 128), 32, 0, stream>>>
    (A_ptr, A_had_ptr, suh_ptr, size_k);
    cuda_check(cudaPeekAtLastError());

    void* args[] = {
        &B_ptr, &C_ptr, &size_m, &size_k, &size_n, &locks_ptr,
        &A_had_ptr, &svh_ptr,
    };
    cuda_check(cudaLaunchCooperativeKernel(
        kernel, dim3(num_sms), dim3(block_dim), args, smem_bytes, stream));
    cuda_check(cudaPeekAtLastError());
}

int exl3_mgemm_bf16_io_grouped_had_m32
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& unique_suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& had_group_ids,
    const at::Tensor& locks,
    int bits,
    int force_num_sms,
    int output_stride,
    bool mcg
)
{
    check_cuda_tensor_m32(A, A, "A");
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    check_cuda_tensor_m32(B, A, "B");
    check_cuda_tensor_m32(C_scratch, A, "C_scratch");
    check_cuda_tensor_m32(C_bf16, A, "C_bf16");
    check_cuda_tensor_m32(unique_suh, A, "unique_suh");
    check_cuda_tensor_m32(A_had, A, "A_had");
    check_cuda_tensor_m32(svh, A, "svh");
    check_cuda_tensor_m32(had_group_ids, A, "had_group_ids");
    check_cuda_tensor_m32(locks, A, "locks");
    TORCH_CHECK_DTYPE(A, kBFloat16);
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(C_scratch, kHalf);
    TORCH_CHECK_DTYPE(C_bf16, kBFloat16);
    TORCH_CHECK_DTYPE(unique_suh, kLong);
    TORCH_CHECK_DTYPE(A_had, kHalf);
    TORCH_CHECK_DTYPE(svh, kLong);
    TORCH_CHECK_DTYPE(had_group_ids, kInt);
    TORCH_CHECK_DTYPE(locks, kInt);
    TORCH_CHECK(mcg, "M32 grouped-Had BF16 MGEMM supports MCG only");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 32,
                "M32 grouped-Had BF16 MGEMM requires A with 32 rows");
    TORCH_CHECK(B.dim() == 1 && C_scratch.dim() == 3 && C_bf16.dim() == 2);
    TORCH_CHECK(A_had.dim() == 3 && svh.dim() == 1 && had_group_ids.dim() == 1);
    TORCH_CHECK(locks.dim() == 1);
    TORCH_CHECK(B.numel() == C_scratch.size(0));
    TORCH_CHECK(C_scratch.size(1) == 32 && C_bf16.size(0) == 32);
    TORCH_CHECK(B.numel() == svh.numel() && B.numel() == had_group_ids.numel());
    TORCH_CHECK(B.numel() > 0, "M32 grouped-Had BF16 MGEMM requires at least one matrix");
    TORCH_CHECK(unique_suh.numel() > 0, "unique_suh must contain at least one group");
    TORCH_CHECK(A_had.size(0) == unique_suh.numel());
    TORCH_CHECK(A_had.size(1) == 32 && A_had.size(2) == A.size(1));
    TORCH_CHECK(output_stride == C_bf16.size(1));
    TORCH_CHECK(output_stride >= B.numel() * C_scratch.size(2));
    TORCH_CHECK(A.size(1) % 128 == 0 && C_scratch.size(2) % 128 == 0,
                "M32 grouped-Had BF16 MGEMM requires K and N divisible by 128");
    const int lock_cols = C_scratch.size(2) / 128;
    TORCH_CHECK(locks.numel() >= B.numel() * lock_cols,
                "locks workspace is too small for grouped MCG reduction");
    if (bits == 5) {
        launch_m32<5, 1>(A, B, C_scratch, C_bf16, unique_suh, A_had,
                         svh, had_group_ids, locks, force_num_sms, output_stride);
        return 2;
    }
    if (bits == 6) {
        launch_m32<6, 1>(A, B, C_scratch, C_bf16, unique_suh, A_had,
                         svh, had_group_ids, locks, force_num_sms, output_stride);
        return 2;
    }
    TORCH_CHECK(false, "M32 grouped-Had BF16 MGEMM supports K5/K6 only");
    return 0;
}

int exl3_gemm_bf16_io_single_had_m32_fp16
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& locks,
    int bits,
    int force_num_sms,
    bool mcg
)
{
    check_cuda_tensor_m32(A, A, "A");
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    check_cuda_tensor_m32(B, A, "B");
    check_cuda_tensor_m32(C, A, "C");
    check_cuda_tensor_m32(suh, A, "suh");
    check_cuda_tensor_m32(A_had, A, "A_had");
    check_cuda_tensor_m32(svh, A, "svh");
    check_cuda_tensor_m32(locks, A, "locks");
    TORCH_CHECK_DTYPE(A, kBFloat16);
    TORCH_CHECK_DTYPE(B, kShort);
    TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK_DTYPE(suh, kHalf);
    TORCH_CHECK_DTYPE(A_had, kHalf);
    TORCH_CHECK_DTYPE(svh, kHalf);
    TORCH_CHECK_DTYPE(locks, kInt);
    TORCH_CHECK(mcg, "M32 single-matrix BF16 MGEMM supports MCG only");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == 32,
                "M32 single-matrix BF16 MGEMM requires A with 32 rows");
    TORCH_CHECK(B.dim() == 3 && C.dim() == 2 && A_had.dim() == 2);
    TORCH_CHECK(suh.dim() == 1 && svh.dim() == 1 && locks.dim() == 1);
    TORCH_CHECK(B.size(0) * 16 == A.size(1),
                "B K dimension does not match A");
    TORCH_CHECK(B.size(1) * 16 == C.size(1),
                "B N dimension does not match C");
    TORCH_CHECK(C.size(0) == 32 && A_had.sizes() == A.sizes());
    TORCH_CHECK(suh.numel() == A.size(1) && svh.numel() == C.size(1),
                "Hadamard scale/flip tensors do not match K/N");
    TORCH_CHECK(A.size(1) % 128 == 0 && C.size(1) % 128 == 0,
                "M32 single-matrix BF16 MGEMM requires K and N divisible by 128");
    TORCH_CHECK(locks.numel() >= C.size(1) / 128,
                "locks workspace is too small for MCG reduction");
    TORCH_CHECK(B.size(2) == bits * 16,
                "trellis storage does not match bits");

    if (bits == 5) {
        launch_single_m32_fp16<5, 1>(
            A, B, C, suh, A_had, svh, locks, force_num_sms);
        return 2;
    }
    if (bits == 6) {
        launch_single_m32_fp16<6, 1>(
            A, B, C, suh, A_had, svh, locks, force_num_sms);
        return 2;
    }
    TORCH_CHECK(false, "M32 single-matrix BF16 MGEMM supports K5/K6 only");
    return 0;
}
