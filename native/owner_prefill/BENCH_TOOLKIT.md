# CUDA toolkit comparison

`bench_toolkit.py` isolates the four owner-prefill extensions using the local
Qwen3.5-27B configuration: TP2, BF16, hidden size 5120, intermediate size 17408,
64 layers, RMSNorm epsilon 1e-6 and weight bias 1. Two SM120 GPUs are required; launch grids adapt to the device SM count.
It checks the supplied model config; it does not load model weights.

Build identical sources with each toolkit, retaining `sm_120f`, fast math and
FlashInfer 0.6.18 headers. The harness changes only operator/C++ namespaces in
its build-directory copies so both variants can coexist. It deliberately
builds both compilers independently of the production wheel's CUDA 13.0 version gate.

```bash
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 \
  python3 native/owner_prefill/bench_toolkit.py build \
  --tag cu130 --out /tmp/mach-owner-ab
CUDA_HOME=/path/to/cuda-13.2 MAX_JOBS=2 \
  python3 native/owner_prefill/bench_toolkit.py build \
  --tag cu132 --out /tmp/mach-owner-ab
CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 native/owner_prefill/bench_toolkit.py run \
  --out /tmp/mach-owner-ab --tags cu132 cu130 \
  --config /data1/models/Qwen3.5-27B-MXFP6/config.json
```

The default cases are M512, M2048, M3000 and M4096, covering both fixed and
ragged kernels. Each tests distributed reduce/residual/RMSNorm, local ordered
sum/residual/RMSNorm, MXFP8 byte/scale gather and BF16 BA byte gather. The final
model-output BF16 gather uses the framework and is outside these extensions.

Each arm has its own collective workspace and shares exactly the same input and output tensors; correctness snapshots are
cloned outside the timed region.
Checks compare the two compilers bitwise on valid owner rows, compare eager
with Graph replay, and change inputs twice before replaying again. Gather
results are also checked against explicit expected rank order and zero padding.
These are synthetic fixtures, not exhaustive real-model numerical acceptance.

Timings use PDL and CUDA Graphs with 100 dispatches per replay, 60 samples per
arm, alternating AB/BA order. Each sample takes the maximum CUDA-event time
across the two ranks; the report gives medians and raw samples. Positive
`candidate_change_pct` means the second tag is slower. Norm outputs are out-of-place. Input/output buffers are
reused, so small cases may benefit from cache residency. There is no GEMM,
attention, scheduler or end-to-end throughput measurement. The script records
compiler versions, flags and input-source hashes in `results.json`.

For a temporary CUDA 13.2.86 toolkit assembled from NVIDIA wheels (without
changing installed packages):

```bash
python3 -m pip install --no-deps --target /tmp/owner-cuda132 \
  nvidia-cuda-nvcc==13.2.86 nvidia-cuda-runtime==13.2.86 \
  nvidia-cuda-crt==13.2.86 nvidia-nvvm==13.2.86 nvidia-cuda-cccl==13.2.86
ln -s libcudart.so.13 /tmp/owner-cuda132/nvidia/cu13/lib/libcudart.so
# Use CUDA_HOME=/tmp/owner-cuda132/nvidia/cu13 for the cu132 build above.
```

## Measured on 2026-09-18

RTX 5090 GPUs 4/5 (same PCIe switch), PyTorch 2.13.0+cu130, FlashInfer 0.6.18; NVCC 13.0.88 versus 13.2.86. Two independent runs reversed compiler load/allocation order. Each used 60 samples per arm and 100 dispatches per Graph replay. All 16 cases passed bitwise compiler, eager/Graph and changing-input checks on both ranks in both runs.

CUDA 13.0 latency change relative to 13.2 (positive = slower):

| Rows | Primitive | Run 1 | Reverse run |
|---:|---|---:|---:|
| 512 | reduce_norm | +1.44% | -0.32% |
| 512 | local_norm | -0.80% | -0.83% |
| 512 | gather_mx8 | -0.47% | +0.47% |
| 512 | gather_ba | -0.92% | +1.05% |
| 2048 | reduce_norm | +0.70% | -0.10% |
| 2048 | local_norm | +0.58% | +0.61% |
| 2048 | gather_mx8 | -0.08% | +0.08% |
| 2048 | gather_ba | -0.89% | +0.91% |
| 3000 | reduce_norm | -0.01% | +0.21% |
| 3000 | local_norm | +2.08% | +1.98% |
| 3000 | gather_mx8 | -0.08% | +0.11% |
| 3000 | gather_ba | -0.16% | +0.10% |
| 4096 | reduce_norm | -0.01% | +0.33% |
| 4096 | local_norm | +1.26% | +1.30% |
| 4096 | gather_mx8 | -0.06% | +0.06% |
| 4096 | gather_ba | -0.50% | +0.57% |

The largest repeatable difference was local norm at M3000: about +2% (roughly 0.28 microseconds per call). M4096 local norm was about +1.3% (0.4 microseconds); communication results were close to parity. This supports adopting CUDA 13.0 with a small measured local-kernel tradeoff, not a claim of zero regression or full-model numerical/performance acceptance. No production kernel algorithm or fast-math flag changed.

Early exploratory measurements used separate output buffers and showed unstable local-norm differences. Those were superseded by the identical-address protocol above. Norm output aliasing, real model captures and end-to-end TTFT/throughput are outside this minimal test.

[Raw samples, compiler flags and source hashes](../../docs/data/owner-prefill-cuda130-20260918.json).

The production `setup.py` was then switched to CUDA 13.0 and successfully built
`vllm_mach_owner_prefill-0.1.0a1-cp312-cp312-linux_x86_64.whl`. All four extensions
loaded with ABI `owner-prefill-v1`; five local row shapes (256, 1024, 1464, 1536,
2048) matched the benchmark build bitwise. The production version gate rejects
CUDA 13.2. The installer was checked to pass one common CUDA_HOME to both
prefill builds.

The broader deployment/install/prefill pytest attempt was interrupted after
24 passes and 10 failures: the host vLLM source did not match the pinned patch
contexts, owner package metadata was absent, and the host lacked
`VLLM_SM120_LOSSLESS_PREFILL_GRAPH`. This is not a successful integration-suite
run. The production wheel was checked from an extracted temporary directory;
the host serving installation was not replaced. Docker was unavailable, so the
updated image recipe was not built here.
