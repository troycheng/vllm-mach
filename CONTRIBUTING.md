# Contributing

Keep changes tied to a concrete serving boundary. A performance change should state the model, checkpoint format, vLLM version, hardware, tensor-parallel layout, request shape, concurrency, graph mode, correctness rule, and the baseline used for comparison.

Correctness comes before performance. Kernel-only measurements support kernel claims; end-to-end throughput or latency claims require a full serving comparison with the same workload and alternating lifecycles where practical. Report unsupported configurations plainly instead of widening the compatibility claim from build success alone.

Use focused commits, add a regression test for behavior changes, and keep private paths, model data, credentials, and internal experiment receipts out of the repository.
