#!/usr/bin/env python3
"""Validate mixed one-shot/twoshot use of a single large FlashInfer workspace.

This is deliberately a no-timing protocol check.  It alternates graph-replayed
small one-shot calls and eager M4096 twoshot calls on each workspace, comparing
the installed FlashInfer reference with the packed M4096 candidate bitwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

import bench_direct_codec as common


HIDDEN = common.HIDDEN
LARGE_MAX_TOKENS = 6553
M4096 = common.ROWS
SMALL_ROWS = (32, 24)
PATTERN = common.PATTERN


@dataclass
class WorkspaceArm:
    name: str
    workspace: Any
    stream: torch.cuda.Stream


@dataclass
class SmallGraph:
    rows: int
    graph: torch.cuda.CUDAGraph
    input: torch.Tensor
    restore: torch.Tensor
    norm_out: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="Exact mach_lossless_prefill_direct extension .so path.")
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def create_workspace(rank: int) -> Any:
    import flashinfer.comm

    workspace = flashinfer.comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=2,
        rank=rank,
        max_token_num=LARGE_MAX_TOKENS,
        hidden_dim=HIDDEN,
        dtype=torch.bfloat16,
        group=dist.group.WORLD,
    )
    if getattr(workspace, "backend", None) != "trtllm":
        raise RuntimeError(f"expected trtllm workspace, got {getattr(workspace, 'backend', None)!r}")
    if not hasattr(workspace, "workspace_tensor") or "buffer_size" not in workspace.metadata:
        raise RuntimeError("workspace lacks workspace_tensor or metadata['buffer_size']")
    if int(workspace.metadata["buffer_size"]) < common.MIN_WORKSPACE_BYTES:
        raise RuntimeError("large workspace does not meet the M4096 twoshot payload/header minimum")
    return workspace


def workspace_metadata(workspace: Any) -> dict[str, Any]:
    return {
        "backend": getattr(workspace, "backend", None),
        "buffer_size": int(workspace.metadata["buffer_size"]),
        "workspace_tensor_shape": list(workspace.workspace_tensor.shape),
        "workspace_tensor_dtype": str(workspace.workspace_tensor.dtype),
        "max_token_num": LARGE_MAX_TOKENS,
        "hidden_dim": HIDDEN,
    }


def deterministic_tensor(rows: int, seed: int, device: torch.device, scale: float = 0.01) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn((rows, HIDDEN), device=device, dtype=torch.float32, generator=generator) * scale).to(torch.bfloat16)


def make_small_inputs(device: torch.device) -> dict[int, torch.Tensor]:
    return {rows: deterministic_tensor(rows, 20261000 + rows, device) for rows in SMALL_ROWS}


def make_residual(rows: int, device: torch.device) -> torch.Tensor:
    return deterministic_tensor(rows, 20261100 + rows, device)


def make_gamma(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20261200)
    return (1.0 + torch.randn((HIDDEN,), device=device, dtype=torch.float32, generator=generator) * 0.01).to(torch.bfloat16)


def installed_dispatch(
    input_: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    workspace: Any,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    use_oneshot: bool,
) -> None:
    import flashinfer.comm

    flashinfer.comm.allreduce_fusion(
        input_,
        workspace=workspace,
        pattern=PATTERN,
        launch_with_pdl=True,
        trigger_completion_at_end=True,
        output=None,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=residual,
        rms_gamma=gamma,
        rms_eps=1e-6,
        use_oneshot=use_oneshot,
        fp32_acc=True,
        weight_bias=1.0,
    )


def candidate_dispatch(
    input_: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    workspace: Any,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    rank: int,
) -> None:
    torch.ops.mach_lossless_prefill_direct.run(
        input_, residual, gamma, workspace.workspace_tensor, residual_out, norm_out,
        rank, int(workspace.metadata["buffer_size"]), 1e-6, 1.0, 1, True,
    )


def capture_small_graph(
    arm: WorkspaceArm,
    source: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
) -> SmallGraph:
    static_input, restore = source.clone(), source.clone()
    norm_out = torch.empty_like(source)
    current = torch.cuda.current_stream(source.device)
    arm.stream.wait_stream(current)
    with torch.cuda.stream(arm.stream):
        installed_dispatch(static_input, residual, gamma, arm.workspace, static_input, norm_out, use_oneshot=True)
    arm.stream.synchronize()
    static_input.copy_(restore)
    arm.stream.wait_stream(torch.cuda.current_stream(source.device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=arm.stream):
        installed_dispatch(static_input, residual, gamma, arm.workspace, static_input, norm_out, use_oneshot=True)
    return SmallGraph(source.shape[0], graph, static_input, restore, norm_out)


def replay_small_graph(arm: WorkspaceArm, graph: SmallGraph) -> tuple[torch.Tensor, torch.Tensor]:
    graph.input.copy_(graph.restore)  # residual_out aliases input; restore outside graph.
    arm.stream.wait_stream(torch.cuda.current_stream(graph.input.device))
    with torch.cuda.stream(arm.stream):
        graph.graph.replay()
    arm.stream.synchronize()
    return graph.input, graph.norm_out


def run_m4096(
    arm: WorkspaceArm,
    source: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    rank: int,
    candidate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    aliased_input = source.clone()
    norm_out = torch.empty_like(source)
    arm.stream.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(arm.stream):
        if candidate:
            candidate_dispatch(aliased_input, residual, gamma, arm.workspace, aliased_input, norm_out, rank)
        else:
            installed_dispatch(aliased_input, residual, gamma, arm.workspace, aliased_input, norm_out, use_oneshot=False)
    arm.stream.synchronize()
    return aliased_input, norm_out


def compare_pair(reference: tuple[torch.Tensor, torch.Tensor], candidate: tuple[torch.Tensor, torch.Tensor], label: str) -> None:
    common.bitwise_equal(reference[0], candidate[0], f"{label}: residual_out/input alias")
    common.bitwise_equal(reference[1], candidate[1], f"{label}: norm_out")


def collect_and_write(
    args: argparse.Namespace,
    rank: int,
    reference: WorkspaceArm,
    candidate: WorkspaceArm,
    records: list[dict[str, Any]],
    fixture_provenance: list[dict[str, Any]],
) -> None:
    payload = {
        "records": records,
        "workspace": {"reference": workspace_metadata(reference.workspace), "candidate": workspace_metadata(candidate.workspace)},
        "fixtures": fixture_provenance,
    }
    gathered: list[Any] | None = [None, None] if rank == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    if rank != 0:
        return
    assert gathered is not None
    summary = {
        "scope": "No-timing mixed protocol validation. No service/GEMM/end-to-end claim.",
        "sequence_per_round": ["M32 one-shot graph", "M4096 twoshot", "M24 one-shot graph", "M4096 twoshot"],
        "rounds": 3,
        "reference": "installed FlashInfer; fp32_acc=True, weight_bias=1.0, launch_with_pdl=True",
        "candidate": "same installed path for M32/M24; packed mach_lossless_prefill_direct twoshot only for M4096, launch_with_pdl=True",
        "alias_contract": "residual_out is input; norm_out is a separate allocation; every graph replay restores its input before launch",
        "deterministic_small_inputs": {"M32_seed": 20261032, "M24_seed": 20261024, "scale": 0.01},
        "deterministic_residuals": {"M32_seed": 20261132, "M24_seed": 20261124, "M4096_seed": 20265196, "scale": 0.01},
        "deterministic_gamma": {"seed": 20261200, "center": 1.0, "stddev": 0.01},
        "library": {"path": str(Path(args.library).resolve()), "sha256": common.sha256_file(Path(args.library))},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "by_rank": gathered,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rank: int | None = None
    reference: WorkspaceArm | None = None
    candidate: WorkspaceArm | None = None
    try:
        rank, _local_rank, device = common.init_dist(args)
        common.create_output_dir(args.output_dir, rank)
        library = Path(args.library)
        if not library.is_file():
            raise FileNotFoundError(f"--library must name an existing .so: {library}")
        torch.ops.load_library(str(library))
        if not hasattr(torch.ops, "mach_lossless_prefill_direct") or not hasattr(torch.ops.mach_lossless_prefill_direct, "run"):
            raise RuntimeError("extension did not register torch.ops.mach_lossless_prefill_direct.run")
        reference = WorkspaceArm("reference", create_workspace(rank), torch.cuda.Stream())
        candidate = WorkspaceArm("candidate", create_workspace(rank), torch.cuda.Stream())
        gamma = make_gamma(device)
        small_inputs = make_small_inputs(device)
        small_residuals = {rows: make_residual(rows, device) for rows in SMALL_ROWS}
        # Captures are per-workspace, always on that workspace's single stream.
        small_graphs = {
            name: {
                rows: capture_small_graph(arm, small_inputs[rows], small_residuals[rows], gamma)
                for rows in SMALL_ROWS
            }
            for name, arm in (("reference", reference), ("candidate", candidate))
        }
        fixture_by_round = [
            common.load_capture_fixture(args.capture_dir, rank, 0, device),
            common.load_capture_fixture(args.capture_dir, rank, 63, device),
            next(fixture for fixture in common.synthetic_fixtures(rank, device) if fixture.name == "zero"),
        ]
        m4096_residual = make_residual(M4096, device)
        records: list[dict[str, Any]] = []
        for round_index, fixture in enumerate(fixture_by_round):
            # The collective ordering is identical on both ranks.  Barriers are not part
            # of a graph and make failures localizable in the short parent-owned run.
            dist.barrier()
            ref_out = replay_small_graph(reference, small_graphs["reference"][32])
            cand_out = replay_small_graph(candidate, small_graphs["candidate"][32])
            compare_pair(ref_out, cand_out, f"round {round_index} M32 graph")
            records.append({"round": round_index, "step": "M32_oneshot_graph", "bitwise": True})

            dist.barrier()
            ref_out = run_m4096(reference, fixture.source, m4096_residual, gamma, rank, candidate=False)
            cand_out = run_m4096(candidate, fixture.source, m4096_residual, gamma, rank, candidate=True)
            compare_pair(ref_out, cand_out, f"round {round_index} M4096 first {fixture.name}")
            records.append({"round": round_index, "step": "M4096_twoshot_first", "fixture": fixture.name, "bitwise": True})

            dist.barrier()
            ref_out = replay_small_graph(reference, small_graphs["reference"][24])
            cand_out = replay_small_graph(candidate, small_graphs["candidate"][24])
            compare_pair(ref_out, cand_out, f"round {round_index} M24 graph")
            records.append({"round": round_index, "step": "M24_oneshot_graph", "bitwise": True})

            dist.barrier()
            ref_out = run_m4096(reference, fixture.source, m4096_residual, gamma, rank, candidate=False)
            cand_out = run_m4096(candidate, fixture.source, m4096_residual, gamma, rank, candidate=True)
            compare_pair(ref_out, cand_out, f"round {round_index} M4096 second {fixture.name}")
            records.append({"round": round_index, "step": "M4096_twoshot_second", "fixture": fixture.name, "bitwise": True})
            if rank == 0:
                print(f"validated round {round_index}: {fixture.name}", flush=True)
        collect_and_write(args, rank, reference, candidate, records, [fixture.provenance for fixture in fixture_by_round])
        dist.barrier()
        if rank == 0:
            print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    finally:
        for arm in (reference, candidate):
            if arm is not None:
                destroy = getattr(arm.workspace, "destroy", None)
                if destroy is not None:
                    try:
                        destroy()
                    except Exception:
                        pass
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"bench_mixed_workspace failed: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
