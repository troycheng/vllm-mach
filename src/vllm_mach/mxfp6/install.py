# SPDX-License-Identifier: Apache-2.0
"""Install the exact native MXFP6 source overlay on a clean target runtime."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
import stat
from contextlib import ExitStack
from pathlib import Path

from vllm_mach.mxfp8.install import (
    _installation_lock,
    _relative,
    _replace,
    apply_profile_patch,
    load_shared_patch,
    sha256,
)

PROFILE = Path(__file__).with_name("profile")
PROFILE_ID = "native-mxfp6-sm120-v033"
_PATCHES = {"vllm": "runtime.patch", "flashinfer": "flashinfer-local-ipc.patch"}


def load_manifest(data_directory: Path | None = None) -> dict:
    directory = PROFILE if data_directory is None else Path(data_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("profile") != PROFILE_ID or manifest.get("schema_version") != 1:
        raise RuntimeError("Unknown native MXFP6 source profile")
    groups = manifest.get("namespaces", {})
    if set(groups) != set(_PATCHES):
        raise RuntimeError("Native MXFP6 requires vllm and flashinfer source namespaces")
    for namespace, patch_name in _PATCHES.items():
        group = groups[namespace]
        if group.get("patch") != patch_name or not group.get("files"):
            raise RuntimeError(f"Invalid {namespace} source manifest")
        for name, entry in group["files"].items():
            _relative(name, namespace)
            for field in ("upstream_sha256", "installed_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", entry.get(field, "")):
                    raise RuntimeError(f"Invalid {field} for {name}")
        if sha256((directory / patch_name).read_bytes()) != group.get("patch_sha256"):
            raise RuntimeError(f"Bundled {namespace} patch SHA256 mismatch")
        if namespace != "vllm" and group.get("shared_patch") is not None:
            raise RuntimeError("Shared runtime patch is only valid in the vllm namespace")
        load_shared_patch(group.get("shared_patch"), group["files"])
    return manifest


def _read_sources(sites: dict[str, Path], manifest: dict) -> tuple[dict, dict]:
    targets, sources = {}, {}
    for namespace, group in manifest["namespaces"].items():
        site = sites[namespace]
        for name in group["files"]:
            relative = _relative(name, namespace)
            target = site / relative
            if any((site / Path(*relative.parts[:end])).is_symlink()
                   for end in range(1, len(relative.parts) + 1)):
                raise RuntimeError(f"Symlinked runtime source: {target}")
            if not target.is_file() or not target.resolve().is_relative_to(site):
                raise RuntimeError(f"Missing or escaped runtime source: {target}")
            targets[name] = target
            sources[name] = target.read_bytes()
    return targets, sources


def _source_state(sources: dict[str, bytes], manifest: dict) -> str:
    states = {"upstream", "installed"}
    for group in manifest["namespaces"].values():
        for name, entry in group["files"].items():
            digest = sha256(sources[name])
            matches = {state for state in states
                       if entry[f"{state}_sha256"] == digest}
            if not matches:
                if digest in (entry["upstream_sha256"], entry["installed_sha256"]):
                    raise RuntimeError("Mixed or partially installed native MXFP6 source profile")
                raise RuntimeError(f"Runtime source SHA256 mismatch: {name}")
            states.intersection_update(matches)
    if len(states) != 1:
        raise RuntimeError("Mixed or empty native MXFP6 source profile")
    return states.pop()


def _commit(sites: dict[str, Path], manifest: dict, targets: dict[str, Path],
            before: dict[str, bytes], after: dict[str, bytes], changes: list[str]) -> None:
    # Lock both distributions in stable order; the shared lock also excludes
    # concurrent MXFP8 publication when both profiles touch the same vLLM tree.
    with ExitStack() as locks:
        for site in sorted(set(sites.values()), key=str):
            locks.enter_context(_installation_lock(site))
        locked_targets, current = _read_sources(sites, manifest)
        if locked_targets != targets or current != before:
            raise RuntimeError("Runtime changed during staging")
        modes = {name: stat.S_IMODE(targets[name].stat().st_mode) for name in changes}
        published = []
        try:
            for name in changes:
                # Include the attempted file: a replacement may publish and
                # then raise (for example a post-write filesystem failure).
                published.append(name)
                _replace(targets[name], after[name], modes[name])
            _, current = _read_sources(sites, manifest)
            if current != after:
                raise RuntimeError("Runtime changed during publication")
        except BaseException as error:
            failures = []
            for name in reversed(published):
                try:
                    _replace(targets[name], before[name], modes[name])
                except BaseException as rollback_error:
                    failures.append(f"{name}: {rollback_error}")
            if failures:
                raise RuntimeError("Runtime install failed and rollback was incomplete: "
                                   + "; ".join(failures)) from error
            raise


def install_profile(site: Path, flashinfer_site: Path, *, apply: bool = False,
                    verify_packages: bool = True,
                    data_directory: Path | None = None) -> dict:
    """Validate the complete pair, stage exact hunks, then publish with rollback.

    Only clean pinned 0.33 upstream or this exact complete installed overlay is
    accepted. Direct offline auditors may disable package checks explicitly;
    the CLI always verifies the central stack and required native producers.
    """
    manifest = load_manifest(data_directory)
    if verify_packages:
        check_versions()
    sites = {"vllm": Path(site).resolve(), "flashinfer": Path(flashinfer_site).resolve()}
    targets, before = _read_sources(sites, manifest)
    state = _source_state(before, manifest)
    directory = PROFILE if data_directory is None else Path(data_directory)
    after = dict(before)
    if state == "upstream":
        for namespace, group in manifest["namespaces"].items():
            sources = {name: before[name] for name in group["files"]}
            after.update(apply_profile_patch(sources, (directory / group["patch"]).read_bytes(),
                                             group.get("shared_patch"), namespace=namespace))
    for group in manifest["namespaces"].values():
        for name, entry in group["files"].items():
            if sha256(after[name]) != entry["installed_sha256"]:
                raise RuntimeError(f"Staged runtime source SHA256 mismatch: {name}")
            compile(after[name], name, "exec")
    changes = [name for name in before if before[name] != after[name]]
    if apply and changes:
        _commit(sites, manifest, targets, before, after, changes)
    return {"profile": manifest["profile"], "vllm": manifest["vllm"],
            "source": manifest["source"], "applied": bool(apply and changes),
            "dry_run": not apply, "changed_files": changes, "previous_state": state,
            "packages_verified": verify_packages,
            "result_sha256": {name: sha256(value) for name, value in after.items()}}


def check_mxfp6_producers() -> None:
    """Reject the older 0.2.1 binary before patching or launching workers."""
    import mxfp6
    import torch

    mxfp6.load_library()
    missing = [
        f"mxfp6.{name}"
        for name in ("gemm_from_swiglu", "gemm_from_gdn")
        if not callable(getattr(mxfp6, name, None))
    ]
    missing.extend(
        f"torch.ops.mxfp6.{name}"
        for name in ("silu_and_mul_mxfp8", "gemm_from_swiglu", "gemm_w6a8_pdl")
        if not hasattr(torch.ops.mxfp6, name)
    )
    if missing:
        raise RuntimeError(
            "MXFP6 is missing required producer operations: " + ", ".join(missing)
            + ". Rebuild Nekofish-L/mxfp6_sm120 at "
            "cd4e964c391fcb8aaf1a27d28a63d778e3a38ece and force-reinstall the wheel; "
            "the original 0.2.1 release is insufficient. See docs/installation.md."
        )


def check_versions() -> None:
    from vllm_mach.stack import BASE_PACKAGES, NATIVE_VERSION, check_versions as check_stack
    check_stack({**BASE_PACKAGES, "mxfp6-sm120": NATIVE_VERSION}, lookup=metadata.version)
    check_mxfp6_producers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="write after all checks pass"
    )
    args = parser.parse_args()
    site = Path(metadata.distribution("vllm").locate_file(""))
    flash_site = Path(metadata.distribution("flashinfer-python").locate_file(""))
    result = install_profile(site, flash_site, apply=args.apply)
    print(json.dumps(result, indent=2))
    if not args.apply:
        print("Preflight passed. Run vllm-mach-install --apply to install.")


if __name__ == "__main__":
    main()
