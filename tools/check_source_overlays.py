#!/usr/bin/env python3
"""Exercise every Mach source installer against pristine vLLM and FlashInfer trees on CPU."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from vllm_mach.fp8 import install as block
from vllm_mach.mxfp6 import install as native
from vllm_mach.mxfp8 import install as mx
from vllm_mach.stack import SUPPORTED_VLLM, profile_directory


def verify(installer, release, vllm_source, flashinfer_source, site):
    is_native = installer is native
    directory = profile_directory(native.PROFILE if is_native else installer.DATA, release)
    manifest = installer.load_manifest(directory)
    groups = manifest["namespaces"] if is_native else {"vllm": manifest}
    source_roots = {"vllm": vllm_source, "flashinfer": flashinfer_source}
    expected = {}
    for namespace, group in groups.items():
        for name, spec in group["files"].items():
            target = site / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_roots[namespace] / name, target)
            expected[name] = spec["installed_sha256"]
    args = (site, site) if is_native else (site,)
    options = {"data_directory": directory, "verify_packages": False}
    preview = installer.install_profile(*args, **options)
    if preview["result_sha256"] != expected or preview["applied"]:
        raise RuntimeError(f"{installer.__name__} returned an invalid preflight")
    installed = installer.install_profile(*args, apply=True, **options)
    verified = installer.install_profile(*args, apply=True, **options)
    if not installed["applied"] or verified["applied"] or verified["result_sha256"] != expected:
        raise RuntimeError(f"{installer.__name__} failed installation or second invocation")
    return {"version": release, "profile": manifest["profile"], "verified_files": len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, choices=SUPPORTED_VLLM)
    parser.add_argument("--vllm-source", required=True, type=Path, help="Directory containing vllm/")
    parser.add_argument("--flashinfer-source", required=True, type=Path, help="Directory containing flashinfer/")
    args = parser.parse_args()
    results = []
    for installer in (mx, block, native):
        with tempfile.TemporaryDirectory() as site:
            results.append(verify(installer, args.version, args.vllm_source, args.flashinfer_source, Path(site)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
