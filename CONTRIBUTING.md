# Contributing

Keep changes tied to a concrete serving boundary. A performance change should state the model, checkpoint format, vLLM version, hardware, tensor-parallel layout, request shape, concurrency, graph mode, correctness rule, and the baseline used for comparison.

Correctness comes before performance. Kernel-only measurements support kernel claims; end-to-end throughput or latency claims require a full serving comparison with the same workload and alternating lifecycles where practical. Report unsupported configurations plainly instead of widening the compatibility claim from build success alone.

Use focused commits, add a regression test for behavior changes, and keep private paths, model data, credentials, and internal experiment receipts out of the repository.

The portable CPU CI checks version admission, build configuration, teacher-token mapping, test-runner contracts and package creation on Python 3.10 and 3.12. Runtime qualification runs inside the matching Mach image with its installed source profile:

```bash
PYTHONPATH=src python3 tools/check_cpu_profile.py --profile mxfp8 --out qualification/mxfp8
```

Use `--profile block-fp8` in the block-FP8 image and `--profile native` in the MXFP6 image. Each run verifies the installed overlay, runs its CPU suite, and writes JSON and JUnit results. The runner isolates tests that replace global module imports and requires the actual-runtime source checks to pass.
