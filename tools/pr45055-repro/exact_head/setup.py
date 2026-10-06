"""Build an exact-PR-head private op without replacing the working vLLM library."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from source_identity import verify_source

HEAD = "42cff10c75958b8cb1ba1cb991ac8bab9f242aa1"
root = Path(os.environ["VLLM_PR45055_SOURCE_ROOT"]).resolve()
source_identity = verify_source(root, os.environ.get("VLLM_PR45055_SOURCE_ARCHIVE"))
source = root / "csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
recorded = Path(__file__).resolve().parents[1] / "upstream/pr45055_42cff10c.cu"
if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(recorded.read_bytes()).digest():
    raise RuntimeError("Exact-head kernel source differs from the downloaded snapshot")
if torch.__version__ != "2.13.0+cu130" or not CUDA_HOME:
    raise RuntimeError("Use the root qualification image Torch2.13.0+cu130/CUDA13.0.88")
compiler = subprocess.check_output([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"],
                                   text=True)
if "release 13.0, V13.0.88" not in compiler:
    raise RuntimeError("CUDA13.0.88 compiler required")
sites = {Path(value) for value in (
    sysconfig.get_path("purelib"), sysconfig.get_path("platlib"), *sys.path
) if value}
runtime = {path.resolve() for site in sites
           for path in site.glob("nvidia/cu13/include")
           if (path / "cuda_fp16.h").is_file() and (path / "cuda_bf16.h").is_file()}
if len(runtime) != 1:
    raise RuntimeError("Expected one pip CUDA13 runtime header directory")
runtime_include, = runtime
system = ["-isystem", str(runtime_include)]
Path("build_identity.json").write_text(json.dumps({
    "head": HEAD, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_identity": source_identity,
    "torch": torch.__version__, "nvcc": compiler,
    "adaptation": "namespace/host identifier rename and private stable registration",
}, indent=2) + "\n")

setup(
    name="pr45055-exact-validation", version="0.0.0",
    ext_modules=[CUDAExtension(
        "pr45055_exact_ext", ["binding.cu"], include_dirs=[str(root / "csrc")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", *system],
            "nvcc": ["-O3", "-std=c++20", "--expt-relaxed-constexpr",
                     "-DTORCH_TARGET_VERSION=0x020B000000000000ULL", "-DUSE_CUDA",
                     "-DENABLE_FP8", "-gencode=arch=compute_120,code=sm_120",
                     "--threads=2", *system],
        },
    )], cmdclass={"build_ext": BuildExtension},
)
