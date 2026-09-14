#!/usr/bin/env python3
"""Apply the owner model hooks after this profile's runtime.patch."""
import argparse
from importlib import metadata, util
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check without writing installed files')
    args = parser.parse_args()
    if metadata.version('vllm') != '0.29.0':
        raise RuntimeError('This owner prefill profile requires vllm==0.29.0')
    spec = util.find_spec('vllm')
    if spec is None or spec.origin is None:
        raise RuntimeError('Cannot locate the installed vLLM package')
    root = Path(spec.origin).resolve().parent.parent
    patch = Path(__file__).with_name('owner-prefill.patch').resolve()
    base = ['patch', '--batch', '--force', '--fuzz=0', '-p1', '-d', str(root), '-i', str(patch)]
    forward = subprocess.run(base + ['--dry-run', '--forward'], capture_output=True, text=True)
    if forward.returncode:
        reverse = subprocess.run(base + ['--dry-run', '--reverse'], capture_output=True, text=True)
        if reverse.returncode == 0:
            print('Owner prefill hooks are already installed')
            return
        raise RuntimeError('Apply the matching runtime.patch first; owner hook context differs:\n' + forward.stdout + forward.stderr)
    if args.check:
        print('Owner prefill hooks can be applied')
        return
    subprocess.run(base + ['--forward'], check=True)
    print('Owner prefill hooks installed; restart vLLM workers to use them')


if __name__ == '__main__':
    main()
