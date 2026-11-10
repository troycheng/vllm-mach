"""Explicit installed-profile selection; generic pytest needs no three-overlay setup."""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITES = json.loads((ROOT / 'tools/cpu_suites.json').read_text())
PROFILES = ('stock', 'mxfp8', 'block-fp8', 'native')


def pytest_addoption(parser):
    parser.addoption('--mach-runtime-profile', choices=PROFILES,
                     default=os.environ.get('MACH_TEST_RUNTIME_PROFILE'),
                     help='Actual installed overlay; omission runs generic tests only')


def pytest_configure(config):
    profile = config.getoption('--mach-runtime-profile')
    if profile is not None and profile not in PROFILES:
        raise pytest.UsageError('Unknown MACH_TEST_RUNTIME_PROFILE: ' + profile)
    if profile is not None:
        os.environ['MACH_TEST_RUNTIME_PROFILE'] = profile


def pytest_ignore_collect(collection_path, config):
    try:
        relative = collection_path.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    allowed = SUITES['profile_files'].get(relative)
    return bool(allowed and config.getoption('--mach-runtime-profile') not in allowed)


def pytest_collection_modifyitems(config, items):
    profile = config.getoption('--mach-runtime-profile')
    for item in items:
        base = item.nodeid.split('[', 1)[0]
        allowed = SUITES['profile_tests'].get(base)
        if allowed and profile not in allowed:
            item.add_marker(pytest.mark.skip(reason='Requires explicit installed runtime profile: ' + ', '.join(allowed)))
