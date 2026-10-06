# Optional community block-FP8 linear kernels

Mach's block-FP8 linear adapter replaces only selected GEMMs in vLLM 0.29.0's `CutlassFp8BlockScaledMMKernel`. Original activation quantization, checkpoint weight bytes, FP32 scales, BF16 output, bias and output reshape are retained. A/B are independently enabled and require CUDA SM120. This adapter accepts community E4M3/FP32 block scales; MXFP8 E8M0 scales use a different backend.

| Route | Qualified tensor boundary | Computation |
| --- | --- | --- |
| n64 Pingpong (A) | N2560, K4096/9216, M16–128 divisible by 8; contiguous input/weight; activation scales column major | Swapped CUTLASS tile 64×32×128, scale granularity 64/1/128, Pingpong, swizzle 1 |
| Ordered low-M (B) | N2560, K4096/9216, M1–8; contiguous inner dimension | Independent K128 FP32 raw-dot partials, ascending-group FP32 scale/FMA reduction, one BF16 conversion |
| Stock fallback | All other shapes, layouts, scale dtypes or output dtypes | Original vLLM CUTLASS behavior |

The n64 route duplicates each original N128 FP32 scale row exactly into two N64 rows. A nonpersistent buffer belongs to each layer and is refreshed by normal `process_weights_after_loading`, including reloads before compilation. No kernel instance stores a last-layer scale pointer. Online weight updates, offload and raw weight/scale copying after graph capture are outside this profile's contract. Discard existing compiled graphs before a reload or disabling the adapter.

Each invocation allocates a fresh output. Ordered raw scratch and native CUTLASS workspace are also independent per invocation, launched on the current stream and owned by the applicable CUDA Graph allocation pool. Outputs and scratch are never cached in process-global storage. Qualified kernel failures propagate; unsupported shapes use the original operation.

The optional `vllm-mach-block-fp8` wheel is built with Torch 2.13.0+cu130, FlashInfer Python 0.6.18 bundled CUTLASS headers and NVCC 13.0.88. No native compilation occurs in the worker or request path. The build discovers the pinned pip CUDA headers using `-isystem` to preserve nvcc's own `crt` search order. Wheel metadata records the source/CUTLASS tree hashes, compiler and Torch ABI. Runtime discovers the installed binary and verifies the exact operator schema. This wheel does not depend on mxfp6, b12x or historical SSM/norm bootstrap code.

```bash
cd native/block_fp8
MAX_JOBS=2 uv pip install --no-build-isolation --no-deps .
```

The FP8 worker lifecycle uses the following API. `register()` can run in the CPU parent and registers only the opaque operator and fake implementation. Verification and installation run after `Worker.init_device`, before model construction; the load hook then prepares each layer's scales without a second model scan.

```python
from vllm_mach.fp8 import linear

linear.register()
# In the GPU worker after init_device:
linear.verify_runtime(n64=True)
linear.install(n64=True, ordered=True)
```

`linear.inspect()` reports eager and capture invocation counts and layer scale memory. It does not count graph replays. `uninstall()` restores the original class methods. Repeated registration/installation is idempotent; changing A/B switches on an installed adapter requires uninstalling and discarding old compiled graphs.

The producer and reducer arithmetic come from the selected September 25 prototype. Current [qualification](fp8-qualification.md) belongs to the four-feature Mach profile: M4/M32/M64 fixed-gold records are byte-identical to stock, and all c4/c16/c32/c64 serving requests succeed under the same 3k/1k protocol. These combined results do not isolate either linear route's end-to-end gain or establish arbitrary-model/context equivalence. Historical gains are not combined with the new measurements.

The CPU suite covers ownership/reload, shape/device/dtype fallback, symbolic M, independent storage and visible runtime failures. The actual-wheel GPU suite passes BF16 byte checks, changing-input separate graphs and separate streams. The tested Python profile is frozen. The final native rebuild after trailing-whitespace-only n64 source cleanup has byte-identical runtime libraries; exact build/source identities are in the [qualification receipt](data/fp8-qualification-20261006.json).
