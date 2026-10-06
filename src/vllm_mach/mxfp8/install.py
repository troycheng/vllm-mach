# SPDX-License-Identifier: Apache-2.0
"""Install the versioned MXFP8 runtime source profile without importing CUDA."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile

PROFILE = "qwen35-4b-mxfp8-champion-v1"
DATA = Path(__file__).with_name("data")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts or path.parts[0] != "vllm"
            or any(part in (".", "..") for part in path.parts)
            or "\\" in name or path.as_posix() != name):
        raise RuntimeError(f"Invalid runtime source path: {name}")
    return Path(*path.parts)


def load_manifest(data_directory: Path | None = None) -> dict:
    directory = DATA if data_directory is None else Path(data_directory)
    manifest = json.loads((directory / "runtime_sources.json").read_text())
    if manifest.get("profile") != PROFILE or manifest.get("schema_version") != 1:
        raise RuntimeError("Unknown MXFP8 runtime source profile")
    for name, entry in manifest["files"].items():
        _relative(name)
        for field in ("upstream_sha256", "installed_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", entry[field]):
                raise RuntimeError(f"Invalid {field} for {name}")
    patch = (directory / "runtime.patch").read_bytes()
    if sha256(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Bundled runtime patch SHA256 mismatch")
    return manifest


def check_packages(expected: dict[str, str]) -> dict[str, str]:
    actual = {name: metadata.version(name) for name in expected}
    for name, required in expected.items():
        if actual[name].split("+", 1)[0] != required:
            raise RuntimeError(f"{PROFILE} requires {name}=={required}, found {actual[name]}")
    return actual


def apply_patch(sources: dict[str, bytes], patch: bytes) -> dict[str, bytes]:
    """Apply UTF-8 unified hunks at their exact offsets, with no fuzz or guessing."""
    lines = patch.decode("utf-8").splitlines(keepends=True)
    result = dict(sources)
    seen = set()
    index = 0
    header = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)\n?$")
    while index < len(lines):
        if not lines[index].startswith("--- a/"):
            raise RuntimeError("Expected unified patch source header")
        name = lines[index][6:].rstrip("\n")
        _relative(name)
        index += 1
        if index >= len(lines) or lines[index] != f"+++ b/{name}\n":
            raise RuntimeError(f"Mismatched patch destination for {name}")
        if name in seen or name not in sources:
            raise RuntimeError(f"Duplicate or untracked patch source: {name}")
        seen.add(name)
        index += 1
        original = sources[name].decode("utf-8").splitlines(keepends=True)
        output = []
        cursor = 0
        hunks = 0
        while index < len(lines) and lines[index].startswith("@@ "):
            match = header.fullmatch(lines[index])
            if not match:
                raise RuntimeError(f"Invalid patch hunk for {name}")
            old_start, old_count, new_start, new_count = match.groups()
            old_count = 1 if old_count is None else int(old_count)
            new_count = 1 if new_count is None else int(new_count)
            start = int(old_start) - (1 if old_count else 0)
            if start < cursor or start > len(original):
                raise RuntimeError(f"Overlapping or out-of-range hunk for {name}")
            output.extend(original[cursor:start])
            if len(output) != int(new_start) - (1 if new_count else 0):
                raise RuntimeError(f"Incorrect destination offset for {name}")
            cursor = start
            index += 1
            removed = added = 0
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- a/")):
                line = lines[index]
                if not line or line[0] not in " +-":
                    raise RuntimeError(f"Unsupported patch line for {name}")
                kind, body = line[0], line[1:]
                if kind in " -":
                    if cursor >= len(original) or original[cursor] != body:
                        raise RuntimeError(f"Exact patch context mismatch in {name}:{cursor + 1}")
                    cursor += 1
                    removed += 1
                if kind in " +":
                    output.append(body)
                    added += 1
                index += 1
            if (removed, added) != (old_count, new_count):
                raise RuntimeError(f"Patch hunk length mismatch for {name}")
            hunks += 1
        if not hunks:
            raise RuntimeError(f"No patch hunks for {name}")
        output.extend(original[cursor:])
        result[name] = "".join(output).encode("utf-8")
    return result


def _read_sources(site: Path, manifest: dict) -> dict[str, bytes]:
    sources = {}
    for name in manifest["files"]:
        target = site / _relative(name)
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f"Missing or symlinked runtime source: {target}")
        if not target.resolve().is_relative_to(site):
            raise RuntimeError(f"Runtime source escapes installation: {target}")
        sources[name] = target.read_bytes()
    return sources


def inspect_sources(site: Path, manifest: dict | None = None) -> dict:
    """Check every tracked source; partial, foreign, and drifted installs fail closed."""
    manifest = load_manifest() if manifest is None else manifest
    site = Path(site).resolve()
    sources = _read_sources(site, manifest)
    states = []
    for name, source in sources.items():
        digest = sha256(source)
        entry = manifest["files"][name]
        if entry["upstream_sha256"] == entry["installed_sha256"]:
            if digest != entry["installed_sha256"]:
                raise RuntimeError(f"Runtime source SHA256 mismatch: {name}")
        elif digest == entry["upstream_sha256"]:
            states.append("upstream")
        elif digest == entry["installed_sha256"]:
            states.append("installed")
        else:
            raise RuntimeError(f"Runtime source SHA256 mismatch: {name}")
    if len(set(states)) != 1:
        raise RuntimeError("Mixed or empty runtime source profile")
    return {"state": states[0], "sha256": {name: sha256(value) for name, value in sources.items()}}


@contextmanager
def _installation_lock(site: Path):
    path = site / ".vllm-mach-mxfp8-install.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Runtime installation is already locked: {path}") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"{os.getpid()}\n")
        yield
    finally:
        path.unlink()


def _replace(target: Path, value: bytes, mode: int) -> None:
    """Publish one complete file on its own filesystem; preserve its permissions."""
    descriptor, name = tempfile.mkstemp(prefix=".mach-mxfp8-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _commit(site: Path, before: dict[str, bytes], after: dict[str, bytes], changes: list[str]) -> None:
    # No source is published until all staged hashes, syntax, and contexts pass.
    with _installation_lock(site):
        for name, source in before.items():
            if (site / name).read_bytes() != source:
                raise RuntimeError(f"Runtime changed during staging: {name}")
        modes = {name: stat.S_IMODE((site / name).stat().st_mode) for name in changes}
        published = []
        try:
            for name in changes:
                _replace(site / name, after[name], modes[name])
                published.append(name)
            for name in after:
                if (site / name).read_bytes() != after[name]:
                    raise RuntimeError(f"Runtime changed during publication: {name}")
        except BaseException as error:
            failures = []
            for name in reversed(published):
                try:
                    _replace(site / name, before[name], modes[name])
                except BaseException as rollback_error:
                    failures.append(f"{name}: {rollback_error}")
            if failures:
                raise RuntimeError("Runtime install failed and rollback was incomplete: " + "; ".join(failures)) from error
            raise


def install_profile(site: Path, *, apply: bool = False, verify_packages: bool = True,
                    data_directory: Path | None = None) -> dict:
    """Dry-run by default. Accept only the pinned upstream or this exact profile.

    ``site`` is the directory containing ``vllm/``. Package verification can be
    disabled only by a direct Python caller doing an offline source audit.
    No MoE, allreduce, IPC, CUDA extension, or model library is built here.
    """
    manifest = load_manifest(data_directory)
    versions = check_packages(manifest["packages"]) if verify_packages else None
    site = Path(site).resolve()
    before = _read_sources(site, manifest)
    state = inspect_sources(site, manifest)["state"]
    directory = DATA if data_directory is None else Path(data_directory)
    after = (apply_patch(before, (directory / "runtime.patch").read_bytes())
             if state == "upstream" else before.copy())
    for name, source in after.items():
        if sha256(source) != manifest["files"][name]["installed_sha256"]:
            raise RuntimeError(f"Staged runtime source SHA256 mismatch: {name}")
        compile(source, name, "exec")
    changes = [name for name in before if before[name] != after[name]]
    if apply and changes:
        _commit(site, before, after, changes)
    return {"profile": PROFILE, "state": state, "applied": bool(apply and changes),
            "dry_run": not apply, "changes": changes, "versions": versions,
            "patch_sha256": manifest["patch_sha256"],
            "result_sha256": {name: sha256(value) for name, value in after.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, help="Directory containing vllm/ (defaults to installed distribution)")
    parser.add_argument("--apply", action="store_true", help="Apply the reviewed source profile; default is dry-run")
    args = parser.parse_args(argv)
    site = args.site or Path(metadata.distribution("vllm").locate_file(""))
    print(json.dumps(install_profile(site, apply=args.apply), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
