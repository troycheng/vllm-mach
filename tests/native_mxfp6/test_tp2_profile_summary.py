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


def test_native_gemm_excludes_bf16_head_and_ba():
    assert summary.category('cutlass::device_kernel<Sm120BlockScaled>') == 'mxfp6_gemm'
    assert summary.category('cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm>') == 'bf16_gemm'
    assert summary.category('cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm>') == 'bf16_gemm'
    assert summary.category('cublasLt::splitKreduce_kernel<32, 16>') == 'bf16_splitk_reduction'
    assert summary.category('fused_recurrent_gated_delta_rule_packed_decode_kernel') == 'gdn_recurrence'
