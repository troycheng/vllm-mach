"""Guard overlap accounting and rejection of mislabeled physical batches."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'tp2_summary', Path(__file__).resolve().parents[2]/'tools/summarize_tp2_decode.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_stream_activity_union():
    assert summary.interval_union([(2,8),(0,5),(8,10),(12,13)]) == 11
    assert summary.interval_union([]) == 0
    with pytest.raises(ValueError):
        summary.interval_union([(2,1)])


def test_memset_not_misidentified_as_gdn_barrier():
    assert summary.category('memset32') == 'buffer_initialization'
    assert summary.category('gdn_fused_decode_kernel') == 'persistent_gdn'
