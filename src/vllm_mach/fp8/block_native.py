# SPDX-License-Identifier: Apache-2.0
"""Load the optional standalone n64 wheel; never compile in a worker."""

from importlib import import_module
from pathlib import Path
import hashlib
import json
import os


_LOADED = False
_IDENTITY = {}
_SCHEMA = "vllm_mach_block_fp8::mm(Tensor a, Tensor b, Tensor sa, Tensor sb) -> Tensor"


def verify_runtime():
    """Verify the installed binary after GPU worker device initialization."""
    global _LOADED, _IDENTITY
    if _LOADED:
        return dict(_IDENTITY)
    torch = import_module("torch")
    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError("block-FP8 native wheel requires torch==2.13.0+cu130")
    override = os.environ.get("VLLM_MACH_BLOCK_FP8_LIBRARY")
    if override:
        library = Path(override).expanduser().resolve()
    else:
        try:
            library = Path(import_module("vllm_mach_block_fp8").library_path()).resolve()
        except ImportError as exc:
            raise ImportError("install the prebuilt vllm-mach-block-fp8 wheel") from exc
    if not library.is_file():
        raise ImportError(f"block-FP8 native library does not exist: {library}")
    torch.ops.load_library(str(library))
    try:
        schema = str(torch.ops.vllm_mach_block_fp8.mm.default._schema)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError("block-FP8 native mm operator is absent") from exc
    if schema != _SCHEMA:
        raise RuntimeError(f"block-FP8 native mm schema mismatch: {schema}")
    metadata_path = library.parent / "build_metadata.json"
    metadata = (json.loads(metadata_path.read_text())
                if metadata_path.is_file() else None)
    if metadata is not None and metadata.get("torch") != torch.__version__:
        raise RuntimeError("block-FP8 binary build metadata has a different Torch ABI")
    _IDENTITY = {"path": str(library),
                 "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                 "torch": torch.__version__, "schema": schema,
                 "build_metadata": metadata}
    _LOADED = True
    return dict(_IDENTITY)


def gemm(a, b, sa, sb64):
    """Invoke the preloaded binary with stock and derived scale layouts."""
    if not _LOADED:
        raise RuntimeError("verify block-FP8 runtime before model compilation")
    return import_module("torch").ops.vllm_mach_block_fp8.mm(a, b.T, sa, sb64.T)
