#!/usr/bin/env python3
"""TP2 M4096 all-reduce/residual/RMSNorm codec boundary screen.

Run under torchrun with exactly two local ranks.  This is a comparison harness,
not a service benchmark: every timed dispatch is the complete M4096 boundary
and uses CUDA-Graph replay, but no scheduler, GEMM, or model work is included.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist


ROWS = 4096
HIDDEN = 5120
SAMPLE_ROWS = 32
REPEAT_ROWS = ROWS // SAMPLE_ROWS
INPUT_BYTES = ROWS * HIDDEN * 2
MIN_WORKSPACE_BYTES = 2 * INPUT_BYTES + (ROWS * HIDDEN // 256) * 2
PATTERN = 1  # flashinfer.comm.AllReduceFusionPattern.kARResidualRMSNorm
CAPTURE_CALLS = (0, 63, 127)
ARM_NAMES = ("installed", "rebuilt_control", "packed")


@dataclass
class Fixture:
    name: str
    source: torch.Tensor
    provenance: dict[str, Any]


@dataclass
class Arm:
    name: str
    workspace: Any
    stream: torch.cuda.Stream
    dispatch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]


@dataclass
class GraphBank:
    graph: torch.cuda.CUDAGraph
    input: torch.Tensor
    restore: torch.Tensor
    residual_out: torch.Tensor
    norm_out: torch.Tensor
    fixture_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="Exact mach_lossless_prefill extension .so path.")
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Run validation and CUDA-Graph replay checks only.")
    parser.add_argument("--smoke", action="store_true", help="Validate synthetic cases only; skip capture fixtures and perf.")
    parser.add_argument("--banks", type=int, default=9, help="Rotating M4096 input/output banks per arm (minimum 8).")
    parser.add_argument("--samples", type=int, default=9, help="Interleaved ABCCBA schedule repetitions.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bitwise_equal(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if not torch.equal(left.view(torch.uint16), right.view(torch.uint16)):
        mismatch = torch.nonzero(left.view(torch.uint16) != right.view(torch.uint16), as_tuple=False)[0]
        where = tuple(int(value) for value in mismatch.tolist())
        raise AssertionError(f"{label}: BF16 bits differ at {where}")


def init_dist(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=dt.timedelta(seconds=args.timeout_seconds),
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"requires torchrun --nproc_per_node=2, got world_size={world_size}")
    if local_rank not in (0, 1):
        raise RuntimeError(f"requires local CUDA devices 0 and 1, got LOCAL_RANK={local_rank}")
    return rank, local_rank, torch.device(f"cuda:{local_rank}")


def create_output_dir(path: Path, rank: int) -> None:
    message: str | None = None
    if rank == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except Exception as error:  # broadcast before raising so peer does not hang at a barrier.
            message = f"--output-dir must not exist: {path} ({error})"
    payload = [message]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(payload[0])
    dist.barrier()


def load_capture_fixture(capture_dir: Path, rank: int, call: int, device: torch.device) -> Fixture:
    stem = f"rank{rank}_call{call:03d}"
    data_path, meta_path = capture_dir / f"{stem}.u16", capture_dir / f"{stem}.json"
    if not data_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing capture pair: {data_path} / {meta_path}")
    metadata = json.loads(meta_path.read_text())
    source_shape = tuple(metadata.get("source_shape", ()))
    if source_shape != (ROWS, HIDDEN):
        raise ValueError(f"{meta_path}: source_shape must be {(ROWS, HIDDEN)}, got {source_shape}")
    raw = np.fromfile(data_path, dtype="<u2")
    if raw.size != SAMPLE_ROWS * HIDDEN:
        raise ValueError(f"{data_path}: expected {SAMPLE_ROWS * HIDDEN} u16 values, got {raw.size}")
    sample_bits = torch.from_numpy(raw.astype(np.uint16, copy=True)).reshape(SAMPLE_ROWS, HIDDEN)
    source = sample_bits.view(torch.bfloat16).repeat((REPEAT_ROWS, 1)).to(device=device, non_blocking=False)
    return Fixture(
        name=f"capture_call{call:03d}",
        source=source,
        provenance={
            "kind": "capture4096_repeat_32x128",
            "rank": rank,
            "call": call,
            "u16": str(data_path),
            "json": str(meta_path),
            "u16_sha256": sha256_file(data_path),
            "json_sha256": sha256_file(meta_path),
            "source_shape": list(source_shape),
        },
    )


def synthetic_fixtures(device: torch.device) -> list[Fixture]:
    zero = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    signed_zero = torch.full((ROWS, HIDDEN), float("-0.0"), dtype=torch.bfloat16, device=device)
    # Fixed CUDA generator makes each rank's residual/norm input independent of host RNG state.
    generator = torch.Generator(device=device).manual_seed(20260907)
    normal = (torch.randn((ROWS, HIDDEN), generator=generator, device=device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return [
        Fixture("zero", zero, {"kind": "synthetic", "bits": "0x0000"}),
        Fixture("signedzero", signed_zero, {"kind": "synthetic", "bits": "0x8000"}),
        Fixture("normal", normal, {"kind": "synthetic", "seed": 20260907, "scale": 0.01}),
    ]


def make_residual_and_gamma(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    residual_generator = torch.Generator(device=device).manual_seed(20260908)
    gamma_generator = torch.Generator(device=device).manual_seed(20260909)
    residual = (torch.randn((ROWS, HIDDEN), generator=residual_generator, device=device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    gamma = (1.0 + torch.randn((HIDDEN,), generator=gamma_generator, device=device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return residual, gamma


def create_workspace(rank: int) -> Any:
    import flashinfer.comm

    workspace = flashinfer.comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=2,
        rank=rank,
        max_token_num=4128,
        hidden_dim=HIDDEN,
        dtype=torch.bfloat16,
        group=dist.group.WORLD,
    )
    if getattr(workspace, "backend", None) != "trtllm":
        raise RuntimeError(f"expected trtllm workspace, got {getattr(workspace, 'backend', None)!r}")
    if not hasattr(workspace, "workspace_tensor") or "buffer_size" not in workspace.metadata:
        raise RuntimeError("FlashInfer workspace does not expose workspace_tensor and metadata['buffer_size']")
    workspace_bytes = int(workspace.metadata["buffer_size"])
    if workspace_bytes < MIN_WORKSPACE_BYTES:
        raise RuntimeError(
            f"workspace buffer_size={workspace_bytes} is below required {MIN_WORKSPACE_BYTES} bytes"
        )
    return workspace


def make_arms(rank: int) -> dict[str, Arm]:
    import flashinfer.comm

    # Each arm owns exactly one workspace and one stream.  Workspaces remain disjoint
    # across installed, rebuilt-control, and packed paths throughout the run.
    installed_ws, rebuilt_ws, packed_ws = (create_workspace(rank) for _ in ARM_NAMES)
    streams = {name: torch.cuda.Stream() for name in ARM_NAMES}

    def installed(input_: torch.Tensor, residual_out: torch.Tensor, norm_out: torch.Tensor) -> None:
        flashinfer.comm.allreduce_fusion(
            input_,
            workspace=installed_ws,
            pattern=PATTERN,
            launch_with_pdl=False,
            trigger_completion_at_end=True,
            output=None,
            residual_out=residual_out,
            norm_out=norm_out,
            residual_in=RESIDUAL,
            rms_gamma=GAMMA,
            rms_eps=1e-6,
            use_oneshot=False,
            fp32_acc=True,
            weight_bias=1.0,
        )

    def extension_dispatch(packed: bool, workspace: Any) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        workspace_bytes = int(workspace.metadata["buffer_size"])

        def dispatch(input_: torch.Tensor, residual_out: torch.Tensor, norm_out: torch.Tensor) -> None:
            torch.ops.mach_lossless_prefill.run(
                input_, RESIDUAL, GAMMA, workspace.workspace_tensor, residual_out, norm_out,
                rank, workspace_bytes, 1e-6, 1.0, packed, False,
            )

        return dispatch

    return {
        "installed": Arm("installed", installed_ws, streams["installed"], installed),
        "rebuilt_control": Arm("rebuilt_control", rebuilt_ws, streams["rebuilt_control"], extension_dispatch(False, rebuilt_ws)),
        "packed": Arm("packed", packed_ws, streams["packed"], extension_dispatch(True, packed_ws)),
    }


# These are initialized after CUDA device selection and are deliberately read-only by every arm.
RESIDUAL: torch.Tensor
GAMMA: torch.Tensor


def run_eager(arm: Arm, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(source)
    norm_out = torch.empty_like(source)
    current = torch.cuda.current_stream(source.device)
    arm.stream.wait_stream(current)
    with torch.cuda.stream(arm.stream):
        arm.dispatch(source.clone(), residual_out, norm_out)
    arm.stream.synchronize()
    return residual_out, norm_out


def run_eager_service_alias(arm: Arm, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the vLLM service alias: residual_out and allreduce input are one tensor."""
    aliased_input = source.clone()
    norm_out = torch.empty_like(source)
    current = torch.cuda.current_stream(source.device)
    arm.stream.wait_stream(current)
    with torch.cuda.stream(arm.stream):
        arm.dispatch(aliased_input, aliased_input, norm_out)
    arm.stream.synchronize()
    return aliased_input, norm_out


def validate_fixtures(arms: dict[str, Arm], fixtures: list[Fixture], rank: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fixture in fixtures:
        dist.barrier()
        outputs = {name: run_eager(arm, fixture.source) for name, arm in arms.items()}
        for name in ("rebuilt_control", "packed"):
            bitwise_equal(outputs["installed"][0], outputs[name][0], f"{fixture.name} residual installed vs {name}")
            bitwise_equal(outputs["installed"][1], outputs[name][1], f"{fixture.name} norm installed vs {name}")
        records.append({"fixture": fixture.name, "provenance": fixture.provenance, "arms": list(ARM_NAMES), "bitwise_pipeline_match": True})
        if rank == 0:
            print(f"validated {fixture.name}: installed == rebuilt_control == packed", flush=True)
    return records


def validate_service_alias(arms: dict[str, Arm], fixtures: list[Fixture], rank: int) -> dict[str, Any]:
    capture = next((fixture for fixture in fixtures if fixture.name.startswith("capture_call")), None)
    if capture is None:
        return {"skipped": "--smoke has no real capture fixture"}
    dist.barrier()
    outputs = {name: run_eager_service_alias(arm, capture.source) for name, arm in arms.items()}
    for name in ("rebuilt_control", "packed"):
        bitwise_equal(outputs["installed"][0], outputs[name][0], f"service-alias residual installed vs {name}")
        bitwise_equal(outputs["installed"][1], outputs[name][1], f"service-alias norm installed vs {name}")
    if rank == 0:
        print("validated service alias: residual_out is allreduce input for real capture", flush=True)
    return {
        "fixture": capture.name,
        "residual_out_aliases_input": True,
        "norm_out_separate": True,
        "bitwise_pipeline_match": True,
    }


def capture_graph(arm: Arm, source: torch.Tensor, fixture_name: str) -> GraphBank:
    static_input = source.clone()
    restore = source.clone()
    # Match vLLM's fused path: all-reduce input storage becomes residual_out.  Every
    # bank owns its own storage and is restored before each replay below.
    residual_out = static_input
    norm_out = torch.empty_like(source)
    current = torch.cuda.current_stream(source.device)
    arm.stream.wait_stream(current)  # make source/full initialization visible before warmup.
    with torch.cuda.stream(arm.stream):
        arm.dispatch(static_input, residual_out, norm_out)
    arm.stream.synchronize()
    static_input.copy_(restore)
    arm.stream.wait_stream(torch.cuda.current_stream(source.device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=arm.stream):
        arm.dispatch(static_input, residual_out, norm_out)
    return GraphBank(graph, static_input, restore, residual_out, norm_out, fixture_name)


def replay_changed_input_graphs(arms: dict[str, Arm], fixtures_by_name: dict[str, Fixture], rank: int) -> list[dict[str, Any]]:
    # Original capture includes the codec's normal compact/raw-fallback mixture; zero exercises
    # the fallback-only shape.  Both reuse graph-static pointers after the input changes.
    original_name = next((name for name in fixtures_by_name if name.startswith("capture_call")), "normal")
    selected = ["zero", original_name]
    graphs = {
        name: capture_graph(arms[name], fixtures_by_name[original_name].source, original_name)
        for name in ("rebuilt_control", "packed")
    }
    records: list[dict[str, Any]] = []
    for fixture_name in selected:
        dist.barrier()  # never inside graph capture or event timing.
        fixture = fixtures_by_name[fixture_name]
        for name, bank in graphs.items():
            current = torch.cuda.current_stream(fixture.source.device)
            bank.input.copy_(fixture.source)
            arms[name].stream.wait_stream(current)
            with torch.cuda.stream(arms[name].stream):
                bank.graph.replay()
            arms[name].stream.synchronize()
        bitwise_equal(graphs["rebuilt_control"].residual_out, graphs["packed"].residual_out, f"graph {fixture_name} residual")
        bitwise_equal(graphs["rebuilt_control"].norm_out, graphs["packed"].norm_out, f"graph {fixture_name} norm")
        records.append({"fixture": fixture_name, "changed_input_graph_replay": True, "bitwise_pipeline_match": True})
        if rank == 0:
            print(f"graph replay validated {fixture_name}: rebuilt_control == packed", flush=True)
    return records


def build_perf_banks(arms: dict[str, Arm], fixtures: list[Fixture], banks: int) -> dict[str, list[GraphBank]]:
    if banks < 8 or banks * INPUT_BYTES < 320 * 1024 * 1024:
        raise ValueError("--banks must provide at least 320 MiB of cold M4096 input rotation (minimum 8)")
    result: dict[str, list[GraphBank]] = {name: [] for name in ARM_NAMES}
    for bank_index in range(banks):
        fixture = fixtures[bank_index % len(fixtures)]
        for name in ARM_NAMES:
            result[name].append(capture_graph(arms[name], fixture.source, fixture.name))
    return result


def time_graph(arm: Arm, bank: GraphBank) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # A graph replay overwrites its input through the service residual alias.  Restore a
    # distinct full M4096 source outside the timed event so a benchmark replay never
    # consumes the previous all-reduce result in place of a fresh GEMM boundary value.
    bank.input.copy_(bank.restore)
    arm.stream.wait_stream(torch.cuda.current_stream(bank.input.device))
    with torch.cuda.stream(arm.stream):
        start.record(arm.stream)
        bank.graph.replay()
        end.record(arm.stream)
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0


def run_perf(arms: dict[str, Arm], fixtures: list[Fixture], banks: int, samples: int) -> list[dict[str, Any]]:
    # Performance uses real-data fixtures only. Every arm in one paired schedule
    # must see exactly the same fixture and bank, including both schedule halves.
    fixtures = [fixture for fixture in fixtures if fixture.name.startswith('capture_call')]
    assert fixtures
    graph_banks = build_perf_banks(arms, fixtures, banks)
    # Symmetric multi-arm ordering reduces drift attribution.  Every rank executes exactly
    # this sequence; Gloo barriers start each group but are outside CUDA event windows.
    schedule = ("installed", "rebuilt_control", "packed", "packed", "rebuilt_control", "installed")
    records: list[dict[str, Any]] = []
    for round_index in range(samples):
        for order_index, arm_name in enumerate(schedule):
            dist.barrier()
            bank_index = round_index % banks
            bank = graph_banks[arm_name][bank_index]
            elapsed_us = time_graph(arms[arm_name], bank)
            records.append({
                "round": round_index,
                "order_index": order_index,
                "arm": arm_name,
                "bank": bank_index,
                "fixture": bank.fixture_name,
                "event_us": elapsed_us,
            })
    return records


def gather_and_write(
    output_dir: Path,
    rank: int,
    args: argparse.Namespace,
    fixtures: list[Fixture],
    validation: list[dict[str, Any]],
    graph_validation: list[dict[str, Any]],
    service_alias_validation: dict[str, Any],
    codec_info: Any,
    perf_records: list[dict[str, Any]],
    device: torch.device,
) -> None:
    gathered: list[Any] | None = [None, None] if rank == 0 else None
    rank_payload = {
        "perf": perf_records,
        "codec_info": codec_info,
        "fixtures": [fixture.provenance | {"name": fixture.name} for fixture in fixtures],
        "validation": validation,
        "graph_validation": graph_validation,
        "service_alias_validation": service_alias_validation,
        "loaded_installed_comm_libraries": {
            str(path): sha256_file(path)
            for path in {Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
                         if line.split() and line.split()[-1].endswith('/trtllm_comm.so')}
        },
    }
    dist.gather_object(rank_payload, gathered, dst=0)
    if rank != 0:
        return
    assert gathered is not None
    rankmax: list[dict[str, Any]] = []
    if perf_records:
        for left, right in zip(gathered[0]["perf"], gathered[1]["perf"], strict=True):
            key = ("round", "order_index", "arm", "bank")
            if tuple(left[field] for field in key) != tuple(right[field] for field in key):
                raise RuntimeError("rank event schedules diverged")
            rankmax.append({
                **{field: left[field] for field in ("round", "order_index", "arm", "bank", "fixture")},
                "rank_event_us": [left["event_us"], right["event_us"]],
                "rankmax_event_us": max(left["event_us"], right["event_us"]),
            })
    props = torch.cuda.get_device_properties(device)
    summary = {
        "scope": "GPU boundary screen only; excludes service scheduling, GEMMs, and end-to-end decode.",
        "configuration": {
            "shape": [ROWS, HIDDEN],
            "dtype": "bfloat16",
            "world_size": 2,
            "pattern": "kARResidualRMSNorm (1)",
            "installed": {"use_oneshot": False, "fp32_acc": True, "weight_bias": 1.0, "launch_with_pdl": False},
            "workspace_min_bytes": MIN_WORKSPACE_BYTES,
            "input_bytes": INPUT_BYTES,
            "input_address_rotation_minimum_bytes": 320 * 1024 * 1024,
            "input_cache_scope": "Each input is restored immediately before its timed boundary and is therefore freshly written, not guaranteed cold in cache. Rotation avoids one fixed input/output address but does not model preceding GEMM weights.",
            "event_scope": "CUDA graph replay only; Gloo barriers and object gather excluded",
        },
        "library": {"path": str(Path(args.library).resolve()), "sha256": sha256_file(Path(args.library))},
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "rank0_gpu": {"name": props.name, "major": props.major, "minor": props.minor, "total_memory": props.total_memory},
        "fixtures_by_rank": [gathered[0]["fixtures"], gathered[1]["fixtures"]],
        "validation_by_rank": [gathered[0]["validation"], gathered[1]["validation"]],
        "changed_input_graph_validation_by_rank": [gathered[0]["graph_validation"], gathered[1]["graph_validation"]],
        "service_alias_validation_by_rank": [gathered[0]["service_alias_validation"], gathered[1]["service_alias_validation"]],
        "codec_info_by_rank": [gathered[0]["codec_info"], gathered[1]["codec_info"]],
        "loaded_installed_comm_libraries_by_rank": [gathered[0]["loaded_installed_comm_libraries"], gathered[1]["loaded_installed_comm_libraries"]],
        "perf_rank0_event_us": gathered[0]["perf"],
        "perf_rank1_event_us": gathered[1]["perf"],
        "perf_rankmax_event_us": rankmax,
        "interpretation": "Only a clear rank-max boundary reduction that exceeds integration risk justifies a complete TP2 AR integration experiment.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def read_codec_info(sample: torch.Tensor) -> Any:
    info = torch.ops.mach_lossless_prefill.info(sample)
    if isinstance(info, torch.Tensor):
        info = info.detach().cpu().tolist()
    values = [int(value) for value in info]
    if len(values) != 7:
        raise RuntimeError(f"mach_lossless_prefill.info expected 7 values, got {values}")
    names = (
        "packed_regs", "packed_local_bytes", "packed_shared_bytes",
        "control_regs", "control_local_bytes", "control_shared_bytes",
        "packed_blocks_per_sm_at_640",
    )
    return dict(zip(names, values, strict=True))


def main() -> None:
    global RESIDUAL, GAMMA
    args = parse_args()
    rank: int | None = None
    arms: dict[str, Arm] | None = None
    try:
        rank, _local_rank, device = init_dist(args)
        create_output_dir(args.output_dir, rank)
        library = Path(args.library)
        if not library.is_file():
            raise FileNotFoundError(f"--library must be an exact existing .so path: {library}")
        torch.ops.load_library(str(library))
        if not hasattr(torch.ops, "mach_lossless_prefill") or not hasattr(torch.ops.mach_lossless_prefill, "run"):
            raise RuntimeError("extension did not register torch.ops.mach_lossless_prefill.run")
        RESIDUAL, GAMMA = make_residual_and_gamma(device)
        fixtures = synthetic_fixtures(device)
        if not args.smoke:
            fixtures.extend(load_capture_fixture(args.capture_dir, rank, call, device) for call in CAPTURE_CALLS)
        codec_info = read_codec_info(fixtures[0].source)
        # All ranks construct workspaces in the same order before issuing any operation.
        arms = make_arms(rank)
        validation = validate_fixtures(arms, fixtures, rank)
        service_alias_validation = validate_service_alias(arms, fixtures, rank)
        fixture_map = {fixture.name: fixture for fixture in fixtures}
        graph_validation = replay_changed_input_graphs(arms, fixture_map, rank)
        perf_records: list[dict[str, Any]] = []
        if not args.validate_only and not args.smoke:
            perf_records = run_perf(arms, fixtures, args.banks, args.samples)
        gather_and_write(
            args.output_dir, rank, args, fixtures, validation, graph_validation,
            service_alias_validation, codec_info, perf_records, device,
        )
        dist.barrier()
        if rank == 0:
            print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    finally:
        if arms is not None:
            for arm in arms.values():
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
        print(f"bench_ar_codec failed: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
