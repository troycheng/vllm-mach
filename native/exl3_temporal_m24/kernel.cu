// MIT-licensed EXL3 Temporal K64, exact-M24 grouped K6 path.
// Device implementation preserved from the validated Temporal-K prototype.
// Reuses ExLlamaV3 fragment, trellis decoder and Hadamard helpers.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
namespace cg = cooperative_groups;
#include "util.h"
#include "util.cuh"
#include "quant/exl3_kernel_map.cuh"
#include "quant/hadamard_inner.cuh"
#include "quant/exl3_gemm_inner.cuh"
#include "bit_windows.cuh"

__global__ void transform(const __nv_bfloat16* a, half* ah,
                          const half** suh, int m, int k) {
    int row = blockIdx.x, col = blockIdx.y * 128, g = blockIdx.z;
    had_bh_r_128_inner<true, false>(a + row * k + col,
        ah + ((size_t)g * m + row) * k + col, suh[g], 0.088388347648f);
}

template<int Bits, int TK, int TM, bool UseWindow, bool HalfAcc = false, bool BankSwizzle = false, bool Grouped = false, int TN = 128, int Stages = 2, int HalfFoldTiles = 4, bool HalfPartial = false>
__global__ __launch_bounds__(TN)
void gemm(const half* ah, const uint16_t** bp, std::conditional_t<HalfPartial,half,float>* partial,
          int m, int k, int n, int splits, const int* had_ids) {
    constexpr int AC = TK / 8;
    constexpr int A_SHIFT = BankSwizzle ? 0 : 1;
    constexpr int AS = TM * TK, BS = TK * TN * Bits / 16;
    extern __shared__ __align__(16) unsigned char raw[];
    half* sa = reinterpret_cast<half*>(raw);
    uint16_t* sb = reinterpret_cast<uint16_t*>(sa + Stages * AS);
    int t = threadIdx.x, warp = t / 32, lane = t % 32;
    int g = blockIdx.z, nt = blockIdx.x, split = blockIdx.y;
    int tiles = k / TK;
    int begin = tiles * split / splits, end = tiles * (split + 1) / splits;
    int a_group = g;
    if constexpr (Grouped) a_group = had_ids[g];
    const half* a = ah + (size_t)a_group * m * k;
    const uint16_t* b = bp[g];
    FragC accum[TM / 16][4] = {};
    FragC_h short_accum[TM / 16][4] = {};
    static_assert(!HalfAcc || TK == 64, "FP16 chunks are four K64 tiles");
    static_assert(!HalfPartial || (HalfAcc && HalfFoldTiles == 0), "compact partials require FP16 sums");

    auto load = [&](int kt, int stage) {
        if (kt < end) {
            // K64/K128 have >=8 int4 columns: row stride no longer supplies
            // alternating bank groups as it does for the original K32 tile.
            #pragma unroll
            for (int i = 0; i < AS / 8; i += TN) {
                int pos = t + i, row = pos / AC, col = pos % AC;
                int4* dst = reinterpret_cast<int4*>(sa + stage * AS) +
                           row * AC + (col ^ ((row >> A_SHIFT) & (AC - 1)));
                if (row < m)
                    cp_async(dst, reinterpret_cast<const int4*>(a + row * k + kt * TK) + col);
                else
                    *dst = make_int4(0, 0, 0, 0);
            }
            #pragma unroll
            for (int i = 0; i < (BS + 7) / 8; i += TN) {
                int pos = t + i;
                if (pos < BS / 8) {
                    constexpr int CHUNK = (TN / 16) * Bits * 2;
                    int kk = pos / CHUNK, nn = pos % CHUNK;
                    auto src = reinterpret_cast<const int4*>(
                        b + ((kt * (TK / 16) + kk) * (n / 16) + nt * (TN / 16)) * Bits * 16);
                    cp_async(reinterpret_cast<int4*>(sb + stage * BS) + pos, src + nn);
                }
            }
        }
        cp_async_fence();
    };

    #pragma unroll
    for (int stage = 0; stage < Stages; ++stage) load(begin + stage, stage);
    for (int kt = begin; kt < end; ++kt) {
        int stage = (kt - begin) % Stages;
        cp_async_wait<Stages - 1>();
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TK / 16; ++kk) {
            FragA fa[TM / 16];
            FragB fb[4];
            int r = (lane % 8) + 8 * ((lane / 8) % 2);
            int col = lane / 16 + kk * 2;
            #pragma unroll
            for (int mm = 0; mm < TM / 16; ++mm) {
                int row = r + mm * 16;
                ldsm4(fa[mm], reinterpret_cast<int4*>(sa + stage * AS) +
                    row * AC + (col ^ ((row >> A_SHIFT) & (AC - 1))));
            }
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const uint32_t* packed = reinterpret_cast<const uint32_t*>(
                    sb + stage * BS + (kk * (TN / 16) + warp * 2 + nn) * Bits * 16);
                if constexpr (UseWindow)
                    dq_dispatch_window<Bits, 1>(packed, lane * 8, fb[nn * 2], fb[nn * 2 + 1]);
                else
                    dq_dispatch<Bits, 1>(packed, lane * 8, fb[nn * 2], fb[nn * 2 + 1]);
            }
            #pragma unroll
            for (int mm = 0; mm < TM / 16; ++mm)
                #pragma unroll
                for (int nn = 0; nn < 4; ++nn)
                    if constexpr (HalfAcc)
                        ptx_mma_m16n8k16(fa[mm], fb[nn], short_accum[mm][nn]);
                    else
                        ptx_mma_m16n8k16(fa[mm], fb[nn], accum[mm][nn]);
        }
        if constexpr (HalfAcc && HalfFoldTiles > 0) {
            // Bound FP16 accumulation to 256 K values, not the entire dot
            // product. The outer sum and cross-CTA partials remain FP32.
            if (((kt - begin + 1) % HalfFoldTiles) == 0 || kt + 1 == end) {
                #pragma unroll
                for (int mm = 0; mm < TM / 16; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < 4; ++nn) {
                        float2 lo = __half22float2(short_accum[mm][nn][0]);
                        float2 hi = __half22float2(short_accum[mm][nn][1]);
                        accum[mm][nn][0] += lo.x; accum[mm][nn][1] += lo.y;
                        accum[mm][nn][2] += hi.x; accum[mm][nn][3] += hi.y;
                        short_accum[mm][nn] = {};
                    }
            }
        }
        // All readers finish before this ring slot is reused. Copies overlap
        // the following tile, not a producer/consumer cross-warp protocol.
        __syncthreads();
        load(kt + Stages, stage);
    }
    cp_async_wait<0>();
    if constexpr (HalfAcc && HalfFoldTiles == 0) {
        // One FP16 accumulator per split; promote once before FP32 reduction.
        #pragma unroll
        for (int mm = 0; mm < TM / 16; ++mm)
            #pragma unroll
            for (int nn = 0; nn < 4; ++nn) {
                float2 lo = __half22float2(short_accum[mm][nn][0]);
                float2 hi = __half22float2(short_accum[mm][nn][1]);
                accum[mm][nn][0] = lo.x; accum[mm][nn][1] = lo.y;
                accum[mm][nn][2] = hi.x; accum[mm][nn][3] = hi.y;
            }
    }
    #pragma unroll
    for (int mm = 0; mm < TM / 16; ++mm)
        #pragma unroll
        for (int nn = 0; nn < 4; ++nn) {
            int row = mm * 16 + lane / 4;
            int col = nt * TN + warp * 32 + nn * 8 + (lane % 4) * 2;
            size_t base = ((size_t)g * splits + split) * m * n;
            auto* out0 = partial + base + row * n + col;
            auto* out1 = out0 + 8 * n;
            if (row < m) {
                if constexpr (HalfPartial)
                    *reinterpret_cast<half2*>(out0) = short_accum[mm][nn][0];
                else { out0[0] = accum[mm][nn][0]; out0[1] = accum[mm][nn][1]; }
            }
            if (row + 8 < m) {
                if constexpr (HalfPartial)
                    *reinterpret_cast<half2*>(out1) = short_accum[mm][nn][1];
                else { out1[0] = accum[mm][nn][2]; out1[1] = accum[mm][nn][3]; }
            }
        }
}

template<typename Partial = float>
__global__ void finish(const Partial* partial, __nv_bfloat16* out,
                       const half** svh, int m, int n, int groups, int splits) {
    __shared__ float sum[4][128];
    int t = threadIdx.x, warp = t / 32, lane = t % 32;
    int col_tile = blockIdx.x % (n / 128);
    int row = (blockIdx.x / (n / 128)) * 4 + warp;
    int col = col_tile * 128, g = blockIdx.z;
    if (row >= m) return; // uniform within each warp; no CTA barrier below
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int c = col + lane + i * 32;
        float v = 0;
        for (int s = 0; s < splits; ++s) {
            auto value = partial[(((size_t)g * splits + s) * m + row) * n + c];
            if constexpr (std::is_same_v<Partial,half>) v += __half2float(value);
            else v += value;
        }
        sum[warp][lane + i * 32] = v;
    }
    __syncwarp();
    had_fhb_r_128_inner<false, true>(sum[warp], out + row * (groups * n) + g * n + col,
                                    svh[g] + col, 0.088388347648f);
}

template<int Bits, int TK, int TM, bool UseWindow, bool HalfAcc = false, bool BankSwizzle = false, bool Grouped = false, int TN = 128, int Stages = 2, int HalfFoldTiles = 4, bool HalfPartial = false>
void launch(const at::Tensor& a, const at::Tensor& b, const at::Tensor& suh,
            const at::Tensor& svh, at::Tensor& ah, at::Tensor& p, at::Tensor& out,
            int splits, cudaStream_t stream, const int* had_ids = nullptr) {
    int m = a.size(0), k = a.size(1), n = p.size(3), groups = b.numel();
    transform<<<dim3(m, k / 128, suh.numel()), 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<half*>(ah.data_ptr()),
        reinterpret_cast<const half**>(suh.data_ptr()), m, k);
    constexpr int smem = Stages * (TM * TK * 2 + TK * TN * Bits / 8);
    using Partial = std::conditional_t<HalfPartial,half,float>;
    gemm<Bits, TK, TM, UseWindow, HalfAcc, BankSwizzle, Grouped, TN, Stages, HalfFoldTiles, HalfPartial><<<dim3(n / TN, splits, groups), TN, smem, stream>>>(
        reinterpret_cast<const half*>(ah.data_ptr()),
        reinterpret_cast<const uint16_t**>(b.data_ptr()), reinterpret_cast<Partial*>(p.data_ptr()), m, k, n, splits, had_ids);
    finish<Partial><<<dim3((n / 128) * ((m + 3) / 4), 1, groups), 128, 0, stream>>>(
        reinterpret_cast<const Partial*>(p.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const half**>(svh.data_ptr()), m, n, groups, splits);
    cuda_check(cudaPeekAtLastError());
}


void temporal_m24(const at::Tensor& a, const at::Tensor& b,
                  const at::Tensor& suh, const at::Tensor& svh,
                  const at::Tensor& ids, at::Tensor& ah,
                  at::Tensor& partial, at::Tensor& out) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    const auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 12 && props->minor == 0, "Temporal M24 requires SM120");
    const at::Tensor* tensors[] = {&b, &suh, &svh, &ids, &ah, &partial, &out};
    for (const auto* t : tensors)
        TORCH_CHECK(t->is_cuda() && t->device() == a.device() && t->is_contiguous());
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
    TORCH_CHECK(b.scalar_type() == at::kLong && suh.scalar_type() == at::kLong &&
                svh.scalar_type() == at::kLong && ids.scalar_type() == at::kInt &&
                ah.scalar_type() == at::kHalf && partial.scalar_type() == at::kFloat);
    TORCH_CHECK(b.dim() == 1 && suh.dim() == 1 && svh.dim() == 1 && ids.dim() == 1);
    const int count = b.numel();
    TORCH_CHECK(a.size(0) == 24 && a.size(1) == 5120);
    TORCH_CHECK(svh.numel() == count && ids.numel() == count &&
                suh.numel() > 0 && suh.numel() <= count);
    TORCH_CHECK(partial.dim() == 4);
    const int splits = partial.size(1), n = partial.size(3);
    TORCH_CHECK((count == 8 && n == 1024 && splits == 10) ||
                (count == 14 && n == 512 && splits == 12), "unsupported Temporal QKV geometry");
    TORCH_CHECK(partial.size(0) == count && partial.size(2) == 24);
    TORCH_CHECK(ah.dim() == 3 && ah.size(0) == suh.numel() &&
                ah.size(1) == 24 && ah.size(2) == 5120);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == 24 && out.size(1) == count * n);
    // IDs and raw weight pointers come from the prevalidated bundle ownership map.
    launch<6,64,32,false,false,true,true>(
        a,b,suh,svh,ah,partial,out,splits,at::cuda::getCurrentCUDAStream().stream(),
        ids.data_ptr<int>());
}
