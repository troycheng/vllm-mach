#!/usr/bin/env python3
"""Qualify one already-installed overlay on CPU. Never installs or patches it."""
from __future__ import annotations
import argparse
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SUITES = json.loads((Path(__file__).with_name('cpu_suites.json')).read_text())
PROFILES = ('mxfp8', 'block-fp8', 'native')


def inspect_profile(profile):
    if profile not in PROFILES:
        raise ValueError(profile)
    from vllm_mach.stack import BASE_PACKAGES, check_versions
    versions = check_versions(BASE_PACKAGES)
    site = Path(metadata.distribution('vllm').locate_file(''))
    if profile == 'native':
        installer = import_module('vllm_mach.mxfp6.install')
        installer.check_versions()  # Includes native producer ABI checks, no GPU work.
        source = installer.install_profile(site, Path(metadata.distribution(
            'flashinfer-python').locate_file('')), verify_packages=False)
        if source['previous_state'] != 'installed' or source['changed_files']:
            raise RuntimeError('Native overlay is not completely installed')
    else:
        installer = import_module('vllm_mach.' + ('fp8' if profile == 'block-fp8' else profile) + '.install')
        source = installer.inspect_sources(site)
        if source['state'] != 'installed':
            raise RuntimeError(profile + ' overlay is not completely installed')
    return {'profile': profile, 'versions': versions, 'vllm_site': str(site),
            'installer': installer.__file__, 'source': source,
            'manifest_sha256': hashlib.sha256((Path(installer.__file__).with_name('profile') / 'manifest.json'
                if profile == 'native' else Path(installer.__file__).with_name('data') / 'runtime_sources.json').read_bytes()).hexdigest()}


def batches(profile):
    result = [('common', SUITES['common'])]
    if SUITES['profile_extra'][profile]:
        result.append(('profile', SUITES['profile_extra'][profile]))
    result.extend((Path(path).stem, [path]) for path in SUITES['isolated_global_module_mocks'])
    return result


def require_sentinels(profile, paths):
    cases = [case for path in paths for case in ET.parse(path).iter('testcase')]
    for required in SUITES['required_tests'][profile]:
        matching = [case for case in cases if required in case.attrib.get('name', '')]
        if not matching or any(any(case.find(tag) is not None for tag in
                                   ('skipped', 'failure', 'error')) for case in matching):
            raise RuntimeError('Required actual-source test did not pass: ' + required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', VLLM_PLUGINS='',
                      MACH_TEST_RUNTIME_PROFILE=args.profile, PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('VLLM_MACH_PROFILE', None)
    receipt = {'profile': args.profile, 'python': sys.executable, 'repo': str(ROOT),
               'cuda_visible_devices': '', 'batches': []}
    try:
        receipt['installed'] = inspect_profile(args.profile)
        xmls = []
        for name, selection in batches(args.profile):
            xml = args.out / (name + '.xml')
            command = [sys.executable, '-m', 'pytest', '-q', *selection,
                       '--mach-runtime-profile=' + args.profile, '--junitxml=' + str(xml)]
            with (args.out / (name + '.log')).open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            receipt['batches'].append({'name': name, 'command': command, 'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError('CPU suite failed: ' + name)
            xmls.append(xml)
        require_sentinels(args.profile, xmls)
        receipt['status'] = 'pass'
    except Exception as error:
        receipt.update(status='fail', error=str(error))
        raise
    finally:
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
