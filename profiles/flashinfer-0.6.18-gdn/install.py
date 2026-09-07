#!/usr/bin/env python3
"""Install the version-locked GDN overlay into a dedicated environment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

PROFILE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def plan(root, profile=PROFILE):
    expected = json.loads((profile / 'base-hashes.json').read_text())
    changes = []
    for relative, original in expected.items():
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts:
            raise RuntimeError(f'Invalid profile path: {relative}')
        source = profile / 'overlay' / name
        destination = root / name
        if not source.is_file():
            raise RuntimeError(f'Missing profile source: {source}')
        if destination.is_symlink() or not destination.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f'Refusing symlink destination: {destination}')
        current, wanted = digest(destination), digest(source)
        if current not in (original, wanted):
            raise RuntimeError(f'Unrecognized installed source: {relative} ({current})')
        if current != wanted:
            changes.append((source, destination))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply after validating every target; default is check only')
    args = parser.parse_args()
    for package, required in [('vllm', '0.28.0'), ('flashinfer-python', '0.6.18')]:
        actual = importlib.metadata.version(package)
        if actual != required:
            raise RuntimeError(f'{package} must be {required}; found {actual}')
    roots = {Path(importlib.metadata.distribution(p).locate_file('')).resolve()
             for p in ('vllm', 'flashinfer-python')}
    if len(roots) != 1:
        raise RuntimeError('vLLM and FlashInfer must be installed in the same environment')
    changes = plan(roots.pop())
    for source, destination in changes:
        print(('install ' if args.apply else 'would install ') + str(destination))
    if args.apply:
        for source, destination in changes:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    print(f'{len(changes)} file(s); restart vLLM before using this profile.')


if __name__ == '__main__':
    main()
