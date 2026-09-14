"""WorkerExtension helpers for the loaded rank64 path; no automatic GPU work.

Call snapshot_worker_inputs(worker) during each of two real M32 decode runs,
while all 32 requests are decoding, then validate_worker with real inputs.
Other graph sizes share the graph pool and may overwrite these intermediate
buffers after requests drain. Verify the capture window against client first-
token and completion times. Without snapshots the optional synthetic mode is
a deterministic kernel/graph check, not model-quality data.
"""
from __future__ import annotations

import torch
import triton as tr
import triton.language as tl


def _states(worker):
    from vllm_mach.exl3.rank64 import ATTRIBUTE
    from vllm_mach.exl3.rank64_assets import MASK
    model = worker.model_runner.model
    states = [getattr(module, ATTRIBUTE) for module in model.modules() if hasattr(module, ATTRIBUTE)]
    if len(states) != 48 or {state.entry["layer"] for state in states} != set(MASK):
        raise ValueError("rank64 probe requires all 48 selected gate/up layers on this worker")
    return sorted(states, key=lambda state: state.entry["layer"])


def snapshot_worker_inputs(worker) -> dict:
    """Capture current M32 intermediates while all 32 real requests decode."""
    states = _states(worker)
    torch.cuda.synchronize()
    captured = []
    for state in states:
        if state.graph_input is None:
            raise RuntimeError("run a real physical-M32 decode before snapshot_worker_inputs")
        value = state.graph_input.detach().clone()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("non-finite graph intermediate: snapshot during steady M32 decode, not after requests drain")
        captured.append((state, value))
    for state, value in captured:
        snapshots = getattr(state, "_probe_inputs", [])
        snapshots.append(value)
        state._probe_inputs = snapshots[-2:]
    torch.cuda.synchronize()
    return {"layers": len(states), "snapshots": min(len(s._probe_inputs) for s in states)}


def _scale_index(device):
    # Independent copy of the reference's labelled inverse of FI interleave.
    from flashinfer.quantization.fp4_quantization import block_scale_interleave
    labels = torch.arange(128 * 320, device=device, dtype=torch.int64).view(128, 320)
    logical = torch.zeros_like(labels)
    for shift in (0, 8, 16):
        logical |= block_scale_interleave(((labels >> shift) & 255).to(torch.uint8)).view(128, 320).long() << shift
    physical = torch.empty_like(labels).flatten()
    physical[logical.flatten()] = torch.arange(128 * 320, device=device)
    return physical.view(128, 320)[:64].contiguous().int()


@tr.jit
def _oracle_stack(X, P, SF, G, INDEX, OUT, M: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    # Source rowstack reference: decode official A1, round residual to BF16,
    # multiply exactly by four, then let the official quantizer create A2.
    i = tl.program_id(0) * B + tl.arange(0, B)
    ok = i < M * K
    x = tl.load(X + i, ok, 0).to(tl.float32)
    packed = tl.load(P + i // 2, ok, 0).to(tl.int32)
    code = (packed >> ((i % 2) * 4)) & 15
    ab = code & 7
    mag = tl.where(ab < 2, ab.to(tl.float32) * 0.5,
                   ((ab & 1) + 2).to(tl.float32) * tl.exp2((ab // 2 - 2).to(tl.float32)))
    signed = tl.where((code & 8) != 0, -mag, mag)
    sf_index = tl.load(INDEX + (i // K) * (K // 16) + (i % K) // 16, ok, 0)
    sf = tl.load(SF + sf_index, ok, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    decoded = signed * sf / tl.load(G)
    residual_bf16 = (x - decoded).to(tl.bfloat16)
    tl.store(OUT + i, x, ok)
    tl.store(OUT + M * K + i, residual_bf16.to(tl.float32) * 4.0, ok)


@tr.jit
def _oracle_merge(TERMS, Z, B, OUT, N: tl.constexpr):
    columns = tl.program_id(0) * 64 + tl.arange(0, 64)
    rows = tl.arange(0, 32)
    ranks = tl.arange(0, 64)
    first = tl.load(TERMS + rows[:, None] * N + columns[None, :]).to(tl.float32)
    second = tl.load(TERMS + (32 + rows[:, None]) * N + columns[None, :]).to(tl.float32)
    z = tl.load(Z + rows[:, None] * 64 + ranks[None, :])
    b = tl.load(B + ranks[:, None] * N + columns[None, :])
    correction = tl.dot(z, b, out_dtype=tl.float32)
    # Retain FP32 addition order and the single final BF16 boundary.
    result = first + second * 0.25 + correction
    tl.store(OUT + rows[:, None] * N + columns[None, :], result)


def _reference(x, state, index):
    import flashinfer
    import vllm._custom_ops as ops
    packed1, scales1 = ops.scaled_fp4_quant(
        x, state.a_global_scale, is_sf_swizzled_layout=True, backend="cutlass")
    stacked = torch.empty((64, 5120), dtype=torch.bfloat16, device=x.device)
    _oracle_stack[(tr.cdiv(x.numel(), 512),)](
        x, packed1, scales1.view(torch.uint8), state.a_global_scale,
        index[:32], stacked, 32, 5120, 512, enable_fp_fusion=False)
    packed, scales = ops.scaled_fp4_quant(
        stacked, state.a_global_scale, is_sf_swizzled_layout=True, backend="cutlass")
    # Serial branch ordering; do not invoke the candidate stream/prepare/merge.
    z = torch.mm(x, state.a)
    terms = flashinfer.gemm.mm_fp4(
        packed, state.packed.T, scales, state.scales.T, state.alpha,
        out_dtype=torch.bfloat16, block_size=16, use_8x4_sf_layout=False,
        backend="b12x", use_nvfp4=True, enable_pdl=False)
    output = torch.empty((32, 17408), dtype=torch.bfloat16, device=x.device)
    _oracle_merge[(17408 // 64,)](terms, z, state.b, output, N=17408,
                                 num_warps=4, enable_fp_fusion=False)
    return packed, scales, output


def validate_worker(worker, require_real_inputs: bool = True) -> dict:
    from vllm_mach.exl3.rank64_assets import tensor_sha256
    states = _states(worker)
    index = _scale_index(states[0].packed.device)
    live = index.long().flatten()
    records = []
    for state in states:
        inputs = getattr(state, "_probe_inputs", [])
        if len(inputs) != 2:
            if require_real_inputs:
                raise ValueError("take two real-request snapshots before validating rank64")
            generator = torch.Generator(device="cpu").manual_seed(20260914 + state.entry["layer"])
            inputs = [torch.randn((32, 5120), generator=generator, dtype=torch.bfloat16).to(state.packed.device) for _ in range(2)]
            source = "synthetic kernel check"
        else:
            source = "real M32 request graph inputs"
        if torch.equal(inputs[0], inputs[1]):
            raise ValueError(f"layer {state.entry['layer']} needs two different input snapshots")
        if not torch.equal(index, state.index):
            raise ValueError("candidate scale index differs from reference FI layout")
        references = []
        for case, x in enumerate(inputs):
            if not bool(torch.isfinite(x).all()):
                raise ValueError(f"layer {state.entry['layer']} case {case}: non-finite snapshot")
            packed, scales, expected = _reference(x, state, index)
            prepared = state._prepare(x, state.a_global_scale, state.index, state.buffers)
            packed_equal = torch.equal(prepared["a_packed"], packed)
            scales_equal = torch.equal(prepared["a_scales"].view(torch.uint8).flatten()[live],
                                       scales.view(torch.uint8).flatten()[live])
            actual = state.apply(x)
            if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise ValueError(f"layer {state.entry['layer']} case {case}: non-finite candidate/reference output")
            records.append({"layer": state.entry["layer"], "rank": state.entry["rank"], "case": case,
                            "input_source": source, "input_sha256": tensor_sha256(x),
                            "packed_equal": packed_equal, "live_scales_equal": scales_equal,
                            "eager_equal": torch.equal(actual, expected),
                            "max_abs": float((actual.float() - expected.float()).abs().max().item())})
            references.append(expected)
        stable = inputs[0].clone()
        state.apply(stable)  # Warm every candidate kernel before capture.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        service_input = state.graph_input
        try:
            with torch.cuda.graph(graph):
                replay_out = state.apply(stable)
        finally:
            state.graph_input = service_input
        for case, x in enumerate(inputs):
            stable.copy_(x)
            graph.replay()
            records[-2 + case]["graph_equal"] = torch.equal(replay_out, references[case])
        del graph
    torch.cuda.synchronize()
    fields = ("packed_equal", "live_scales_equal", "eager_equal", "graph_equal")
    return {"passed": all(all(row[field] for field in fields) for row in records),
            "layers": len(states), "cases": len(records), "checks": len(records) * len(fields),
            "real_inputs_required": require_real_inputs, "records": records}
