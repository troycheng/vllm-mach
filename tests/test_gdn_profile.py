import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'profiles' / 'flashinfer-0.6.18-gdn'
spec = importlib.util.spec_from_file_location('mach_gdn_installer', PROFILE / 'install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def fixture_profile(tmp_path):
    profile = tmp_path / 'profile'
    (profile / 'overlay' / 'pkg').mkdir(parents=True)
    (profile / 'overlay' / 'pkg' / 'a.py').write_text('new\n')
    (profile / 'base-hashes.json').write_text(json.dumps({'pkg/a.py':None}))
    root = tmp_path / 'installed'
    root.mkdir()
    return root, profile


def test_new_and_idempotent(tmp_path):
    root, profile = fixture_profile(tmp_path)
    changes = installer.plan(root, profile)
    assert len(changes) == 1
    source, destination = changes[0]
    destination.parent.mkdir()
    destination.write_bytes(source.read_bytes())
    assert installer.plan(root, profile) == []


def test_unknown_source_rejected_without_writes(tmp_path):
    root, profile = fixture_profile(tmp_path)
    (root / 'pkg').mkdir()
    target = root / 'pkg' / 'a.py'
    target.write_text('unrelated')
    with pytest.raises(RuntimeError, match='Unrecognized'):
        installer.plan(root, profile)
    assert target.read_text() == 'unrelated'


def test_profile_complete_and_opt_in():
    files = json.loads((PROFILE / 'base-hashes.json').read_text())
    assert len(files) == 12
    assert all((PROFILE / 'overlay' / f).is_file() for f in files)
    gdn = (PROFILE / 'overlay/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py').read_text()
    assert 'os.environ.get("VLLM_ENABLE_QWEN_GDN_FUSED_DECODE", "0")' in gdn


def test_symlink_rejected(tmp_path):
    root, profile = fixture_profile(tmp_path)
    (root / 'pkg').mkdir()
    (root / 'pkg' / 'a.py').symlink_to(profile / 'overlay/pkg/a.py')
    with pytest.raises(RuntimeError, match='symlink'):
        installer.plan(root, profile)
