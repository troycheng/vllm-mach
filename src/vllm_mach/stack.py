# SPDX-License-Identifier: Apache-2.0
"""CPU-only version contract shared by runtime profiles and build tooling."""
from importlib import metadata

MACH_VERSION = "0.3.0"
BASE_PACKAGES = {
    "vllm": "0.33.0", "torch": "2.14.0", "triton": "3.8.0",
    "flashinfer-python": "0.7.2.post1", "flashinfer-cubin": "0.7.2.post1",
    "nvidia-cutlass-dsl": "4.8.0", "cuda-bindings": "13.4.4",
}
TORCH_BUILD = "2.14.0+cu130"
BUILD_PACKAGES = {"cmake": "4.1.0", "b12x": "1.5.1"}
NATIVE_VERSION = "0.3.0"
NATIVE_EXTENSIONS = {
    "vllm-mach-ar-norm": "0.2.0",
    "vllm-mach-owner-prefill": "0.2.0",
    "vllm-mach-lossless-prefill": "0.2.0",
    "vllm-mach-block-fp8": "0.2.0",
    "vllm-mach-fp8-activation": "0.2.0",
}
CACHE_NAMESPACE = f"mach-{MACH_VERSION}-vllm{BASE_PACKAGES['vllm']}-torch{TORCH_BUILD}"


def require_version(name, found, expected):
    """Keep Torch's build suffix: CPU/other-CUDA wheels are not this ABI."""
    target = TORCH_BUILD if name == "torch" else expected
    actual = found if name == "torch" else found.split("+", 1)[0]
    if actual != target:
        raise RuntimeError(f"Mach requires {name}=={target}; found {found}")
    return found


def check_versions(packages, *, lookup=None):
    lookup = metadata.version if lookup is None else lookup
    actual = {}
    for name, expected in packages.items():
        try:
            found = lookup(name)
        except metadata.PackageNotFoundError as exc:
            target = TORCH_BUILD if name == "torch" else expected
            raise RuntimeError(f"Mach requires installed {name}=={target}") from exc
        actual[name] = require_version(name, found, expected)
    return actual
