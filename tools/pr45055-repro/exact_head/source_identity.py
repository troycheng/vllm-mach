"""Verify a pristine exact-head checkout or an unchanged official archive tree."""

import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

HEAD = "42cff10c75958b8cb1ba1cb991ac8bab9f242aa1"


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_source(root, archive=None):
    root = Path(root).resolve()
    if archive is None:
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != HEAD:
            raise RuntimeError(f"Expected exact public PR head {HEAD}; found {actual}")
        if subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip():
            raise RuntimeError("The exact-head source checkout has modifications")
        result = {"mode": "git", "verified_git_head": actual}
    else:
        archive = Path(archive).resolve()
        prefix = "vllm-" + HEAD
        if root.name != prefix:
            raise RuntimeError(f"Archive source root must have prefix {prefix}")
        entries = {}
        with tarfile.open(archive, mode="r|*") as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise RuntimeError(f"Unsafe archive path: {member.name}")
                if not name.parts or name.parts[0] != prefix:
                    raise RuntimeError(f"Unexpected archive prefix: {member.name}")
                relative = PurePosixPath(*name.parts[1:]).as_posix()
                if relative == ".":
                    if not member.isdir():
                        raise RuntimeError("Archive root is not a directory")
                    continue
                if relative in entries:
                    raise RuntimeError(f"Duplicate archive entry: {relative}")
                target = root / relative
                if member.isdir():
                    if target.is_symlink() or not target.is_dir():
                        raise RuntimeError(f"Source directory differs: {relative}")
                    entries[relative] = {"type": "directory"}
                elif member.issym():
                    if not target.is_symlink() or target.readlink().as_posix() != member.linkname:
                        raise RuntimeError(f"Source symlink differs: {relative}")
                    entries[relative] = {"type": "symlink", "target": member.linkname}
                elif member.isfile():
                    stream = tar.extractfile(member)
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                    if target.is_symlink() or not target.is_file() or digest_file(target) != expected:
                        raise RuntimeError(f"Source file differs: {relative}")
                    entries[relative] = {"type": "file", "sha256": expected}
                else:
                    raise RuntimeError(f"Unsupported archive entry: {relative}")
        actual_entries = {path.relative_to(root).as_posix() for path in root.rglob("*")}
        if actual_entries != entries.keys():
            difference = sorted(actual_entries.symmetric_difference(entries))
            raise RuntimeError(f"Source/archive entries differ: {difference[:8]}")
        result = {
            "mode": "archive", "declared_public_head": HEAD,
            "archive_path": str(archive), "archive_sha256": digest_file(archive),
            "archive_prefix": prefix, "entries": entries,
            "verification": "archive/source entry types, symlinks and every file SHA256",
        }
    source = root / "csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
    recorded = Path(__file__).resolve().parents[1] / "upstream/pr45055_42cff10c.cu"
    if digest_file(source) != digest_file(recorded):
        raise RuntimeError("Exact-head kernel differs from the independent API snapshot")
    result["kernel_sha256"] = digest_file(source)
    return result
