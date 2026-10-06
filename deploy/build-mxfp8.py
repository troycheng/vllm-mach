#!/usr/bin/env python3
"""Build the declared official MXFP8/dual source against the base Torch ABI.

Use --source-dir for a local source audit/build or --revision for a full official
commit. --check-only validates local sources without CUDA, downloads or builds.
The source wheel's standard setup.py builds both official CMake targets.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import zipfile

from install import patch_file

REPOSITORY = "Nekofish-L/mxfp6_sm120"
CUTLASS = "e6233cbac5d7c7a865c19c91cd684ceece19513c"
PATCHES = (
    "0001-sm120-mxfp6-small-tile-runtime.patch",
    "0003-sm120-streamk-persistent-workspace.patch",
    "0004-sm120-single-stage-mainloop.patch",
    "0005-sm120-static-problem-shape.patch",
)
BASE_PACKAGES = {"vllm": "0.29.0", "torch": "2.13.0", "triton": "3.7.1",
                 "flashinfer-python": "0.6.18", "flashinfer-cubin": "0.6.18",
                 "nvidia-cutlass-dsl": "4.6.2", "cuda-bindings": "13.3.1"}
EXCLUDED_DIRECTORIES = {".git", "build", "dist", "__pycache__", ".pytest_cache", ".cache", "CMakeFiles", "_skbuild", "third_party"}
EXCLUDED_FILES = {"CMakeCache.txt", "build.ninja", ".ninja_log", ".ninja_deps"}
EXCLUDED_SUFFIXES = {".so", ".o", ".a", ".whl", ".pyc", ".pyo", ".cubin", ".ptx"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(source: Path) -> dict[str, str]:
    result = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (any(part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts)
                or path.name in EXCLUDED_FILES or path.suffix in EXCLUDED_SUFFIXES or not path.is_file()):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Source symlinks are not admitted: {relative}")
        result[relative.as_posix()] = digest(path)
    return result


def validate_source(source: Path) -> dict:
    """Reject old MXFP6 revisions that do not implement all three dual variants."""
    required = ["CMakeLists.txt", "setup.py", "pyproject.toml", "python/mxfp6/mxfp8.py",
                "python/mxfp6/mxfp8_dual.py", "csrc/mxfp8_dual_extension.cu",
                "csrc/mxfp8_dual_mlp32.cu", "csrc/mxfp8_dual_qkvz32.cu",
                "csrc/mxfp8_dual_qkvz64_pipereg.cu"]
    required += ["patches/cutlass/" + name for name in PATCHES]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError("MXFP8/dual source is incomplete; missing: " + ", ".join(missing))
    cmake = (source / "CMakeLists.txt").read_text()
    setup = (source / "setup.py").read_text()
    for target in ("mxfp6_torch", "mxfp8_torch"):
        if f"add_library({target}" not in cmake or target not in setup:
            raise RuntimeError(f"Standard CMake/wheel build lacks target {target}")
    for name in ("mxfp8_dual_mlp32.cu", "mxfp8_dual_qkvz32.cu", "mxfp8_dual_qkvz64_pipereg.cu"):
        if name not in cmake:
            raise RuntimeError(f"Standard CMake target does not compile {name}")
    if "dual_gemm_out" not in (source / "csrc/mxfp8_dual_extension.cu").read_text():
        raise RuntimeError("Official dual_gemm_out dispatcher is missing")
    return {"targets": ["mxfp6_torch", "mxfp8_torch"],
            "patch_sha256": {name: digest(source / "patches/cutlass" / name) for name in PATCHES},
            "source_sha256": source_hashes(source)}


def archive(repository: str, revision: str, parent: Path) -> tuple[Path, dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Supply a full 40-character source commit; tags and branch names are not pins")
    parent.mkdir(parents=True, exist_ok=False)
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
    tarball = parent / "source.tar.gz"
    urllib.request.urlretrieve(url, tarball)
    with tarfile.open(tarball) as stream:
        stream.extractall(parent, filter="data")
    source = parent / (repository.split("/")[-1] + "-" + revision)
    if not source.is_dir():
        raise RuntimeError("Downloaded source archive has an unexpected root")
    return source, {"repository": repository, "revision": revision,
                    "archive_sha256": digest(tarball)}


def copy_source(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes(source)
    for name in hashes:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    if source_hashes(destination) != hashes:
        raise RuntimeError("Source changed while making the clean build snapshot")
    receipt = {"kind": "explicit-source-directory", "source_sha256": hashes}
    if (source / ".git").exists():
        revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"],
                                check=True, capture_output=True, text=True).stdout
        receipt.update(git_revision=revision, git_dirty=bool(status))
    return receipt


def discover_toolchain(cuda_home: Path | None = None, site_paths: list[Path] | None = None) -> dict:
    cuda = cuda_home or Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0"))
    cuda = cuda.resolve()
    nvcc = cuda / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA compiler not found: {nvcc}")
    libraries = sorted((cuda / "targets/x86_64-linux/lib").glob("libnvrtc.so.13.0.*"))
    if not libraries:
        libraries = sorted((cuda / "lib64").glob("libnvrtc.so.13.0.*"))
    libraries = sorted({path.resolve() for path in libraries if path.is_file()})
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one CUDA 13.0 NVRTC library, found: {libraries}")
    if site_paths is None:
        site_paths = [Path(value) for value in {sysconfig.get_path("purelib"),
                      sysconfig.get_path("platlib"), *sys.path} if value]
    includes = {path.resolve() for site in site_paths for path in site.glob("nvidia/cu13/include")
                if (path / "cuda_fp16.h").is_file() and (path / "cuda_bf16.h").is_file()}
    if len(includes) != 1:
        raise RuntimeError(f"Expected one installed nvidia/cu13/include directory, found: {sorted(includes)}")
    include, = includes
    return {"cuda_home": str(cuda), "nvcc": str(nvcc), "nvrtc": str(libraries[0]),
            "cu13_include": str(include)}


def cmake_arguments(cutlass: Path, toolchain: dict) -> list[str]:
    return [f"-DCUTLASS_DIR={cutlass}", f"-DCMAKE_CUDA_COMPILER={toolchain['nvcc']}",
            f"-DCUDA_NVRTC_LIB={toolchain['nvrtc']}",
            "-DCMAKE_CUDA_FLAGS=-isystem " + shlex.quote(toolchain["cu13_include"])]


def versions() -> dict[str, str]:
    result = {name: metadata.version(name) for name in BASE_PACKAGES}
    for name, expected in BASE_PACKAGES.items():
        if result[name].split("+", 1)[0] != expected:
            raise RuntimeError(f"Public base must retain {name}=={expected}; found {result[name]}")
    for name, expected in {"cmake": "4.1.0", "b12x": "1.2.6"}.items():
        result[name] = metadata.version(name)
        if result[name] != expected:
            raise RuntimeError(f"Build requires {name}=={expected}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path, help="Explicit official source tree; snapshot without binaries/cache")
    source.add_argument("--revision", help="Full official source commit containing dual MLP32/QKVZ32/QKVZ64")
    parser.add_argument("--work-dir", type=Path, default=Path("/opt/mach-build"))
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--max-jobs", type=int, default=int(os.environ.get("MAX_JOBS", "2")))
    parser.add_argument("--check-only", action="store_true", help="Only validate --source-dir; no network/CUDA/build")
    args = parser.parse_args(argv)
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    if args.check_only:
        if args.source_dir is None:
            parser.error("--check-only requires --source-dir")
        print(json.dumps(validate_source(args.source_dir.resolve()), indent=2))
        return 0
    before_versions = versions()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if args.source_dir is not None:
        kernel = work / "kernel"
        kernel_receipt = copy_source(args.source_dir.resolve(), kernel)
    else:
        downloaded, kernel_receipt = archive(REPOSITORY, args.revision, work / "kernel-archive")
        kernel = work / "kernel"
        snapshot = copy_source(downloaded, kernel)
        kernel_receipt.update(kind="official-commit-archive", source_sha256=snapshot["source_sha256"])
    checked = validate_source(kernel)
    cutlass, cutlass_receipt = archive("NVIDIA/cutlass", CUTLASS, work / "cutlass-archive")
    for name in PATCHES:
        patch_file(cutlass, kernel / "patches/cutlass" / name)
    toolchain = discover_toolchain(args.cuda_home)
    compiler = subprocess.run([toolchain["nvcc"], "--version"], check=True,
                              capture_output=True, text=True).stdout
    if "release 13.0" not in compiler:
        raise RuntimeError("Build requires the public base CUDA 13.0 compiler")
    toolchain.update(nvcc_version=compiler.strip(),
                     nvcc_sha256=digest(Path(toolchain["nvcc"])),
                     nvrtc_sha256=digest(Path(toolchain["nvrtc"])))
    cmake = cmake_arguments(cutlass, toolchain)
    env = dict(os.environ, CUDA_HOME=toolchain["cuda_home"], CUDACXX=toolchain["nvcc"],
               TORCH_CUDA_ARCH_LIST="12.0a", MAX_JOBS=str(args.max_jobs),
               CMAKE_ARGS=shlex.join(cmake))
    env["PATH"] = str(Path(toolchain["nvcc"]).parent) + os.pathsep + env["PATH"]
    wheels = work / "wheels"
    wheels.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    str(kernel), "-w", str(wheels)], env=env, check=True)
    wheel, = wheels.glob("mxfp6_sm120-0.2.1-*.whl")
    with zipfile.ZipFile(wheel) as package:
        for name in ("mxfp6/mxfp6_torch.so", "mxfp6/mxfp8_torch.so", "mxfp6/mxfp8_dual.py"):
            if name not in package.namelist():
                raise RuntimeError(f"Wheel does not contain the required target/API: {name}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
    # Dispatcher registration and shape selection require no CUDA allocations.
    probe = """import json, torch, mxfp6
from mxfp6 import mxfp8, mxfp8_dual
mxfp6.load_library()
a=mxfp8.load_library(); b=mxfp8_dual.load_library()
assert a == b
for shape in mxfp8_dual.SUPPORTED_SHAPES: assert mxfp8_dual.select_variant(*shape)
print(json.dumps({'library_sha256': __import__('hashlib').sha256(a.read_bytes()).hexdigest(),
 'schemas': {n:str(getattr(torch.ops.mxfp8_sm120,n).default._schema) for n in ('gemm','allocate_workspace','dual_gemm_out')}}))
"""
    probe_result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if probe_result.returncode:
        raise RuntimeError("Native dispatcher registration failed:\n" + probe_result.stdout + probe_result.stderr)
    after_versions = versions()
    if before_versions != after_versions:
        raise RuntimeError("Kernel wheel changed the pinned public base package versions")
    receipt = {"kernel": kernel_receipt, "cutlass": cutlass_receipt, **checked,
               "cutlass_patched_source_sha256": source_hashes(cutlass),
               "toolchain": toolchain, "cmake_arguments": cmake, "max_jobs": args.max_jobs,
               "wheel": {"name": wheel.name, "sha256": digest(wheel)},
               "python": sys.version, "versions": after_versions, "mxfp6-sm120": metadata.version("mxfp6-sm120"),
               "dispatcher": json.loads(probe_result.stdout)}
    (work / "mxfp8-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
