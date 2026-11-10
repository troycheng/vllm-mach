#!/usr/bin/env python3
"""Compatibility entry point for the unified MXFP6/MXFP8 wheel builder."""
import importlib.util
from pathlib import Path
import sys

# Public native 0.3.0 source shared with the Docker builds.
REVISION = "1aa0955aa0bedf6f2bbc87be367301fe5f7c4c8b"

_spec = importlib.util.spec_from_file_location(
    "mach_unified_native_builder", Path(__file__).with_name("build-mxfp8.py")
)
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)

# Retain the old import helpers for callers, with one implementation/pin source.
CUTLASS = _builder.CUTLASS
PATCHES = _builder.PATCHES
discover_toolchain = _builder.discover_toolchain
cmake_arguments = _builder.cmake_arguments

SMOKE = """import torch, mxfp6
from mxfp6 import mxfp8
mxfp6.load_library()
mxfp8.load_library()
assert callable(mxfp6.gemm_from_gdn)
assert hasattr(torch.ops.mxfp6, 'gemm_from_swiglu')
assert hasattr(torch.ops.mxfp6, 'gemm_w6a8_pdl')
ba = str(torch.ops.mxfp8_sm120.bf16_ba_m16.default._schema)
assert 'Tensor x' in ba and 'Tensor weight' in ba, ba
gemm = str(torch.ops.mxfp8_sm120.gemm.default._schema)
assert 'bool use_pdl=False' in gemm, gemm
"""


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == flag or arg.startswith(flag + "=")
               for arg in args for flag in ("--revision", "--source-dir")):
        args = ["--revision", REVISION, *args]
    # The shared builder emits mxfp8-build.json containing both native DSOs.
    return _builder.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
