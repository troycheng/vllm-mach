# SPDX-License-Identifier: Apache-2.0
"""Stage and install the pinned block-FP8 source seams; CPU only."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import re

# Shared exact-hunk staging/rollback mechanics, not the MXFP8 source profile.
from ..mxfp8.install import (
    _commit, _read_sources, _relative, apply_patch, inspect_sources as inspect_manifest,
    sha256,
)
from .profile import NAME

DATA = Path(__file__).with_name("data")


def load_manifest(data_directory=None):
    directory = DATA if data_directory is None else Path(data_directory)
    manifest = json.loads((directory / "runtime_sources.json").read_text())
    if manifest.get("profile") != NAME or manifest.get("schema_version") != 1:
        raise RuntimeError("Unknown block-FP8 source profile")
    for name, entry in manifest["files"].items():
        _relative(name)
        for field in ("upstream_sha256", "installed_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", entry[field]):
                raise RuntimeError(f"Invalid {field} for {name}")
    patch = (directory / "runtime.patch").read_bytes()
    if sha256(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Block-FP8 patch SHA256 differs")
    return manifest


def inspect_sources(site, manifest=None):
    return inspect_manifest(Path(site), load_manifest() if manifest is None else manifest)


def install_profile(site, *, apply=False, verify_packages=True, data_directory=None):
    directory = DATA if data_directory is None else Path(data_directory)
    manifest = load_manifest(directory)
    versions = None
    if verify_packages:
        versions = {name: metadata.version(name) for name in manifest["packages"]}
        for name, required in manifest["packages"].items():
            if versions[name].split("+", 1)[0] != required:
                raise RuntimeError(f"Block-FP8 requires {name}=={required}; found {versions[name]}")
    site = Path(site).resolve()
    before = _read_sources(site, manifest)
    state = inspect_sources(site, manifest)["state"]
    after = (apply_patch(before, (directory / "runtime.patch").read_bytes())
             if state == "upstream" else before.copy())
    for name, source in after.items():
        if sha256(source) != manifest["files"][name]["installed_sha256"]:
            raise RuntimeError(f"Staged block-FP8 source differs: {name}")
        compile(source, name, "exec")
    changes = [name for name in before if before[name] != after[name]]
    if apply and changes:
        _commit(site, before, after, changes)
    return {"profile": NAME, "state": state, "applied": bool(apply and changes),
            "dry_run": not apply, "changes": changes, "versions": versions,
            "patch_sha256": manifest["patch_sha256"],
            "result_sha256": {name: sha256(value) for name, value in after.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    site = args.site or Path(metadata.distribution("vllm").locate_file(""))
    print(json.dumps(install_profile(site, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
