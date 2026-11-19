# SPDX-License-Identifier: Apache-2.0
"""CPU-only version contract shared by runtime profiles and build tooling."""
from importlib import metadata
import ast

MACH_VERSION = "0.3.0"
SUPPORTED_VLLM = ("0.33.0", "0.34.0")
_PACKAGES_033 = {
    "vllm": "0.33.0", "torch": "2.14.0", "triton": "3.8.0",
    "flashinfer-python": "0.7.2.post1", "flashinfer-cubin": "0.7.2.post1",
    "nvidia-cutlass-dsl": "4.8.0", "cuda-bindings": "13.4.4",
}
_PACKAGES_034 = {
    **_PACKAGES_033, "vllm": "0.34.0",
    "flashinfer-python": "0.7.3", "flashinfer-cubin": "0.7.3",
    "cuda-bindings": "13.5.0",
}


def packages_for_vllm(release):
    if release == "0.33.0":
        return _PACKAGES_033.copy()
    if release == "0.34.0":
        return _PACKAGES_034.copy()
    raise RuntimeError(f"Unsupported vLLM version {release}. Supported versions: {', '.join(SUPPORTED_VLLM)}")


try:
    _installed_vllm = metadata.version("vllm").split("+", 1)[0]
except metadata.PackageNotFoundError:
    # Offline source inspection and native wheel builds can run before vLLM is installed.
    _installed_vllm = "0.33.0"

# An unrelated vLLM installation can discover the plugin and decline registration.
BASE_PACKAGES = packages_for_vllm(_installed_vllm) if _installed_vllm in SUPPORTED_VLLM else _PACKAGES_033.copy()


def profile_directory(root, vllm_version=None):
    """Return the pinned overlay files for the selected upstream release."""
    release = _installed_vllm if vllm_version is None else vllm_version
    if release == "0.33.0":
        return root
    if release == "0.34.0":
        return root / "v034"
    raise RuntimeError(f"Unsupported vLLM version {release}. Supported versions: {', '.join(SUPPORTED_VLLM)}")


TORCH_BUILD = "2.14.0+cu130"
BUILD_PACKAGES = {"cmake": "4.1.0", "b12x": "1.5.1"}
NATIVE_VERSION = "0.3.1"
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


def _installed_torch_build():
    """Read the wheel's generated identity without loading libtorch or CUDA."""
    source = metadata.distribution("torch").locate_file("torch/version.py")
    for node in ast.parse(source.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__version__"):
            return ast.literal_eval(node.value)
    raise RuntimeError("Torch wheel has no generated build identity")


def check_versions(packages, *, lookup=None):
    lookup = metadata.version if lookup is None else lookup
    actual = {}
    for name, expected in packages.items():
        try:
            found = lookup(name)
        except metadata.PackageNotFoundError as exc:
            target = TORCH_BUILD if name == "torch" else expected
            raise RuntimeError(f"Mach requires installed {name}=={target}") from exc
        if name == "torch" and found == _PACKAGES_034["torch"] and _installed_vllm == "0.34.0":
            # The upstream 0.34 wheel's METADATA omits the cu130 suffix.
            found = _installed_torch_build()
        actual[name] = require_version(name, found, expected)
    return actual
