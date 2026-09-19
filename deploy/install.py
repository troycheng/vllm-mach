#!/usr/bin/env python3
"""Build optional prefill extensions and install the native MXFP6 profile."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def patch_file(target, patch):
    """Apply a pinned external source patch once, with zero fuzz."""
    command = [
        "patch",
        "--batch",
        "--force",
        "--fuzz=0",
        "-p1",
        "-d",
        str(target),
        "-i",
        str(patch),
    ]
    if (
        subprocess.run(
            command + ["--dry-run", "--reverse"], capture_output=True
        ).returncode
        == 0
    ):
        return
    run(*command, "--dry-run", "--forward")
    run(*command, "--forward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-home", type=Path,
        help="CUDA 13.0 toolkit for both prefill extensions",
    )
    parser.add_argument(
        "--lossless-cuda-home", type=Path,
        help="Optional lossless toolkit path override (CUDA 13.0)",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/mach-build"))
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--native-only", action="store_true")
    phases.add_argument("--runtime-only", action="store_true")
    parser.add_argument(
        "--native-part",
        choices=("lossless_prefill", "owner_prefill", "ar_norm"),
    )
    args = parser.parse_args()
    if not args.runtime_only:
        parts = (
            [args.native_part]
            if args.native_part
            else ["lossless_prefill", "owner_prefill", "ar_norm"]
        )
        wheels = args.work_dir / "prefill-wheels"
        wheels.mkdir(parents=True, exist_ok=True)
        for part in parts:
            cuda = (
                (args.lossless_cuda_home or args.cuda_home)
                if part == "lossless_prefill"
                else args.cuda_home
            )
            if cuda is None:
                parser.error(
                    "Supply --cuda-home pointing to a CUDA 13.0 toolkit"
                )
            env = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST="12.0a")
            env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
            run(
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                ROOT / "native" / part,
                "-w",
                wheels,
                env=env,
            )
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            *sorted(wheels.glob("*.whl")),
        )
    if args.native_only:
        return
    run(sys.executable, "-m", "pip", "install", "--no-deps", str(ROOT))
    run(sys.executable, "-m", "vllm_mach.mxfp6.install", "--apply")


if __name__ == "__main__":
    main()
