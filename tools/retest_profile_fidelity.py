"""Run current Dense or MoE fidelity profiles against matched BF16 references."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "dense": {
        "bf16": "Qwen3.8-27B-official",
        "default": "Qwen3.8-27B-MXFP6",
        "full": "Qwen3.8-27B-MXFP6",
        "fp8": "Qwen3.8-27B-FP8-official",
        "nvfp4": "Qwen3.8-27B-NVFP4",
    },
    "moe": {
        "bf16": "Qwen3.5-35B-A3B",
        "default": "Qwen3.5-35B-A3B-MXFP6",
        "full": "Qwen3.5-35B-A3B-MXFP6",
        "fp8": "Qwen3.5-35B-A3B-FP8",
        "nvfp4": "Qwen3.5-35B-A3B-NVFP4-ScaleSweep",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--models", type=Path, default=Path("/data1/models"))
    parser.add_argument(
        "--stock-pythonpath",
        type=Path,
        required=True,
        help="Unpatched official vLLM/FlashInfer module directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--physical-rows", nargs="+", type=int, choices=(4, 32), default=(32, 4)
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=MODELS["dense"],
        default=("default", "full", "bf16", "fp8", "nvfp4"),
    )
    args = parser.parse_args()
    for package in ("vllm", "flashinfer"):
        if not (args.stock_pythonpath / package / "__init__.py").is_file():
            parser.error(f"Missing stock package: {args.stock_pythonpath / package}")
    root = args.output.resolve() / args.family
    root.mkdir(parents=True, exist_ok=True)
    models = MODELS[args.family]
    for rows in args.physical_rows:
        for arm in args.arms:
            dest = root / f"m{rows}" / arm
            # Never mix new and partial/previous runs implicitly.
            if dest.exists():
                raise FileExistsError(f"Use a new output directory: {dest}")
            paths = [str(ROOT / "src")]
            if arm in ("bf16", "fp8", "nvfp4"):
                paths.insert(0, str(args.stock_pythonpath.resolve()))
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=args.devices,
                OMP_NUM_THREADS="4",
                PYTHONPATH=os.pathsep.join(paths),
            )
            # A partially matching shared MoE tactic cache can make one
            # rank skip tuning while another waits in a tuning collective.
            if args.family == "moe":
                env["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = str(
                    root / f"autotune-m{rows}-{arm}"
                )
            command = [
                sys.executable,
                str(ROOT / "tools/fidelity_native_mxfp6.py"),
                "--arm",
                arm,
                "--current-profile",
                "--model",
                str(args.models / models[arm]),
                "--tokenizer",
                str(args.models / models["bf16"]),
                "--manifest",
                str(ROOT / "docs/data/fidelity-samples.json"),
                "--output",
                str(dest),
                "--physical-rows",
                str(rows),
                "--skip-head-probe",
                "--cpu-offload-gb",
                "12" if args.family == "moe" else "4",
            ]
            (root / f"m{rows}-{arm}-launch.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "environment": {
                            k: env[k]
                            for k in (
                                "CUDA_VISIBLE_DEVICES",
                                "OMP_NUM_THREADS",
                                "PYTHONPATH",
                                "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                            )
                            if k in env
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            print("START", args.family, rows, arm, flush=True)
            with (root / f"m{rows}-{arm}.log").open("x") as log:
                subprocess.run(
                    command,
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print("COMPLETE", args.family, rows, arm, flush=True)


if __name__ == "__main__":
    main()
