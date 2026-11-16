"""CPU entrypoint selection and mandatory execution contracts, without vLLM."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cpu_profile_entry', ROOT / 'tools/check_cpu_profile.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_profile_mapping_is_disjoint_and_preserves_native_source_check():
    for profile in entry.PROFILES:
        plan = entry.batches(profile)
        assert len(plan) == (5 if profile != 'block-fp8' else 4)
        common = plan[0][1]
        assert not set(common) & set(entry.SUITES['isolated_global_module_mocks'])
        isolated = [paths[0] for _, paths in plan if len(paths) == 1
                    and paths[0] in entry.SUITES['isolated_global_module_mocks']]
        assert isolated == entry.SUITES['isolated_global_module_mocks']
    assert entry.SUITES['profile_tests'][
        'tests/native_mxfp6/test_moe_quant_admission_cpu.py::test_installed_native_constructor_matches_audited_fixture'] == ['native']


@pytest.mark.parametrize('bad', ['skipped', 'failure', 'error', 'missing'])
def test_selected_source_sentinel_cannot_be_missing_or_skipped(tmp_path, bad):
    suite = ET.Element('testsuite')
    for name in entry.SUITES['required_tests']['native']:
        if bad == 'missing' and 'constructor' in name:
            continue
        case = ET.SubElement(suite, 'testcase', name=name)
        if 'constructor' in name and bad != 'missing':
            ET.SubElement(case, bad)
    path = tmp_path / 'results.xml'
    ET.ElementTree(suite).write(path)
    with pytest.raises(RuntimeError, match='actual-source test did not pass'):
        entry.require_sentinels('native', [path])
    for case in suite:
        for child in list(case):
            case.remove(child)
    if bad == 'missing':
        ET.SubElement(suite, 'testcase', name=entry.SUITES['required_tests']['native'][1])
    ET.ElementTree(suite).write(path)
    entry.require_sentinels('native', [path])
