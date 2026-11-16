"""Selected-profile qualification must inspect the real installed source tree."""
import importlib.util
import os
from pathlib import Path


def test_selected_runtime_profile_sources_are_installed():
    path = Path(__file__).resolve().parents[1] / 'tools/check_cpu_profile.py'
    spec = importlib.util.spec_from_file_location('mach_cpu_profile_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.inspect_profile(os.environ['MACH_TEST_RUNTIME_PROFILE'])
    assert result['profile'] == os.environ['MACH_TEST_RUNTIME_PROFILE']
