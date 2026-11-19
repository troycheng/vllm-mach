#!/usr/bin/env python3
"""Install the locked build and runtime dependencies for an official vLLM image."""
import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index-url")
    source.add_argument("--wheel-dir", type=Path)
    args = parser.parse_args()
    release = metadata.version("vllm").split("+", 1)[0]
    pip = [sys.executable, "-m", "pip"]
    directory = Path(__file__).parent
    if release == "0.33.0":
        requirements = directory / "requirements-mxfp8.txt"
        # The 0.33 image mixes these 1.45-era fragments with the 1.40 family.
        subprocess.run(pip + ["uninstall", "-y", "opentelemetry-exporter-otlp-common",
                              "opentelemetry-exporter-http-transport"], check=True)
    elif release == "0.34.0":
        requirements = directory / "requirements-vllm034.txt"
        # This optional external KV connector pins telemetry <=1.40, while
        # vLLM 0.34 requires >=1.45.1. Mach's local KV cache is built into vLLM.
        try:
            lmcache = metadata.version("lmcache")
        except metadata.PackageNotFoundError:
            lmcache = None
        if lmcache == "0.5.7.post1":
            subprocess.run(pip + ["uninstall", "-y", "lmcache"], check=True)
    else:
        parser.error(f"Unsupported vLLM base {release}")
    flags = []
    if args.wheel_dir:
        flags = ["--no-index", "--find-links", str(args.wheel_dir)]
    elif args.index_url:
        flags = ["--index-url", args.index_url]
    subprocess.run(pip + ["install", "--no-deps", "--require-hashes", *flags, "-r", str(requirements)], check=True)
    subprocess.run(pip + ["check"], check=True)


if __name__ == "__main__":
    main()
