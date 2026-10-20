# SPDX-License-Identifier: Apache-2.0
"""The migrated runtime must pass both ordered-GDN source guards."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("worker", ["gdn", "two_b.gdn"])
def test_ordered_registration_accepts_current_source_and_rejects_drift(worker):
    program = f"""
import importlib
module = importlib.import_module("vllm_mach.mxfp8.{worker}.worker")
expected = module.GDN_SHA
module.GDN_SHA = "0" * 64
try:
    module._register_gdn()
except RuntimeError as error:
    assert "live GDN source SHA drift" in str(error)
else:
    raise AssertionError("foreign source accepted")
module.GDN_SHA = expected
module._register_gdn()
module._register_gdn()
"""
    subprocess.run([sys.executable, "-c", program], check=True)
