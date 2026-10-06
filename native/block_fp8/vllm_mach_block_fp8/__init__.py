# SPDX-License-Identifier: Apache-2.0
"""Binary discovery without importing Torch or initializing CUDA."""

from pathlib import Path


def library_path() -> Path:
    libraries = sorted(Path(__file__).parent.glob("_C*.so"))
    if len(libraries) != 1:
        raise ImportError("expected one installed vllm-mach-block-fp8 CUDA library")
    return libraries[0]
