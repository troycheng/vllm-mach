# SPDX-License-Identifier: Apache-2.0
"""Install the native MXFP6 source profile into an official vLLM 0.29.0 wheel."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

PROFILE = Path(__file__).with_name("profile")


def _patch(root: Path, name: str, *, reverse: bool = False, dry: bool = False):
    command = [
        "patch",
        "--batch",
        "--force",
        "--fuzz=0",
        "-p1",
        "-d",
        str(root),
        "-i",
        str(PROFILE / name),
    ]
    command += ["--reverse" if reverse else "--forward"]
    if dry:
        command.append("--dry-run")
    return subprocess.run(command, capture_output=True, text=True)


def install_profile(site: Path, flashinfer_site: Path, *, apply: bool = False) -> dict:
    """Stage all patches before writing; reject incomplete/incompatible patches."""
    manifest = json.loads((PROFILE / "manifest.json").read_text())
    files = manifest["files"] + manifest["moe_source"]["files"]
    for name in files:
        target = site / name
        if not target.is_file():
            raise RuntimeError(f"Missing official vLLM file: {target}")
    flash_files = ["flashinfer/comm/mnnvl.py", "flashinfer/comm/trtllm_ar.py"]
    with tempfile.TemporaryDirectory(prefix="mach-mxfp6-") as tmp:
        stage = Path(tmp)
        targets = {name: site / name for name in files}
        targets.update({name: flashinfer_site / name for name in flash_files})
        for name, target in targets.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, destination)
        # Peel additive compilation support before checking the base profile.
        if (
            _patch(stage, "compiled-ar-norm.patch", reverse=True, dry=True).returncode
            == 0
        ):
            result = _patch(stage, "compiled-ar-norm.patch", reverse=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        # Peel the additive MoE fusion patch before checking the base profile.
        # This also permits atomic upgrades from dense-only / initial MoE installs.
        if _patch(stage, "moe-ar-norm.patch", reverse=True, dry=True).returncode == 0:
            result = _patch(stage, "moe-ar-norm.patch", reverse=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        installed = (
            _patch(stage, "runtime.patch", reverse=True, dry=True).returncode == 0
        )
        state = "installed" if installed else "upstream"
        if not installed:
            result = _patch(stage, "runtime.patch")
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        # Separate patch permits upgrading an already installed dense profile.
        if _patch(stage, "moe.patch", reverse=True, dry=True).returncode != 0:
            result = _patch(stage, "moe.patch")
            if result.returncode:
                raise RuntimeError(
                    "MXFP6 MoE patch failed: " + result.stdout + result.stderr
                )
        result = _patch(stage, "moe-ar-norm.patch")
        if result.returncode:
            raise RuntimeError(
                "MXFP6 MoE AR/Norm patch failed: " + result.stdout + result.stderr
            )
        result = _patch(stage, "compiled-ar-norm.patch")
        if result.returncode:
            raise RuntimeError(
                "MXFP6 compiled AR/Norm patch failed: " + result.stdout + result.stderr
            )
        ipc_installed = (
            _patch(
                stage, "flashinfer-local-ipc.patch", reverse=True, dry=True
            ).returncode
            == 0
        )
        if not ipc_installed:
            result = _patch(stage, "flashinfer-local-ipc.patch")
            if result.returncode:
                raise RuntimeError(
                    "FlashInfer IPC patch failed: " + result.stdout + result.stderr
                )
        changes = [
            name
            for name, target in targets.items()
            if target.read_bytes() != (stage / name).read_bytes()
        ]
        if apply:
            # Preserve exact bytes for rollback if any write fails. No files are
            # touched until all dependency profiles pass their staged checks.
            originals = {name: targets[name].read_bytes() for name in changes}
            written = []
            try:
                for name in changes:
                    written.append(name)
                    targets[name].write_bytes((stage / name).read_bytes())
            except BaseException:
                for name in reversed(written):
                    targets[name].write_bytes(originals[name])
                raise
    return {
        "profile": "native-mxfp6",
        "vllm": manifest["vllm"],
        "source": manifest["source"],
        "applied": apply,
        "changed_files": changes,
        "previous_state": state,
    }


def check_versions() -> None:
    for name, expected in (
        ("vllm", "0.29.0"),
        ("torch", "2.13.0"),
        ("flashinfer-python", "0.6.18"),
        ("flashinfer-cubin", "0.6.18"),
        ("nvidia-cutlass-dsl", "4.6.2"),
        ("mxfp6-sm120", "0.2.1"),
    ):
        found = metadata.version(name).split("+", 1)[0]
        if found != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {found}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="write after all checks pass"
    )
    args = parser.parse_args()
    check_versions()
    site = Path(metadata.distribution("vllm").locate_file(""))
    flash_site = Path(metadata.distribution("flashinfer-python").locate_file(""))
    result = install_profile(site, flash_site, apply=args.apply)
    print(json.dumps(result, indent=2))
    if not args.apply:
        print("Preflight passed. Run vllm-mach-install --apply to install.")


if __name__ == "__main__":
    main()
