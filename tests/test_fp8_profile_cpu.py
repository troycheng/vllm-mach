"""Public profile selection and source-install isolation without CUDA."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from vllm_mach.fp8 import profile, serve


def test_launch_keeps_full_capacity_precision_and_independent_features():
    with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
        os.environ["VLLM_MACH_LM_HEAD_SAMPLING"] = "1"
        first = serve.configure_environment(temporary, features=("ordered", "fa2"))
        assert first["VLLM_MACH_LM_HEAD_SAMPLING"] == "0"
        assert profile.selected_features() == ("ordered", "fa2")
        assert first["VLLM_MACH_FP8_N64"] == first["VLLM_MACH_FP8_SILU"] == "0"
        assert first["VLLM_MACH_FP8_COMPILE_MODE"] == "production"
        assert first["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        second = serve.configure_environment(temporary, features=("n64",))
        assert profile.selected_features() == ("n64",)
        assert first["VLLM_CACHE_ROOT"] != second["VLLM_CACHE_ROOT"]
        assert Path(first["VLLM_CACHE_ROOT"]).parent.is_dir()
    args = serve.build_argv("/models/block", tokenizer="/models/tokenizer")
    expected = {"--max-num-seqs": "128", "--max-num-batched-tokens": "2048",
                "--max-model-len": "8192", "--kv-cache-memory": str(19 * 2**30),
                "--mamba-ssm-cache-dtype": "float32", "--dtype": "bfloat16",
                "--linear-backend": "cutlass", "--attention-backend": "FLASH_ATTN",
                "--attention-config": '{"flash_attn_version":2}',
                "--additional-config": '{"gdn_prefill_backend":"triton"}',
                "--tokenizer": "/models/tokenizer"}
    assert all(args[args.index(key) + 1] == value for key, value in expected.items())


@pytest.mark.parametrize("features", [(), ("n64", "n64"), ("unknown",)])
def test_bad_feature_selection_has_no_environment_or_directory_side_effect(features):
    with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
        target = Path(temporary) / "absent"
        with pytest.raises(ValueError):
            serve.configure_environment(target, features=features)
        assert not target.exists() and not os.environ


def test_profile_refuses_combined_lossy_flags_or_dependency_drift():
    settings = {"VLLM_MACH_PROFILE": profile.NAME, "VLLM_MACH_FP8_N64": "1"}
    with patch.dict(os.environ, settings, clear=True):
        actual = {**profile.DEPENDENCIES, "torch": "2.14.0+cu130"}
        with patch.object(profile, "version", side_effect=actual.__getitem__):
            assert profile.check_environment() == actual
            os.environ["VLLM_HYBRID_NVFP4_LM_HEAD"] = "1"
            with pytest.raises(RuntimeError, match="cannot be combined"):
                profile.check_environment()
        del os.environ["VLLM_HYBRID_NVFP4_LM_HEAD"]
        with patch.object(profile, "version", return_value="0.30.0"):
            with pytest.raises(RuntimeError, match="requires"):
                profile.check_environment()


def test_nested_weight_offload_configs_require_a_different_profile():
    from vllm_mach.fp8.worker import validate_worker_config

    cfg = NS(
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           decode_context_parallel_size=1, enable_dbo=False),
        speculative_config=None,
        additional_config={"gdn_prefill_backend": "triton"},
        model_config=NS(dtype="torch.bfloat16", quantization="compressed-tensors",
            hf_text_config=NS(model_type="qwen3_5_text", hidden_size=2560,
                intermediate_size=9216, num_hidden_layers=32, num_attention_heads=16,
                num_key_value_heads=4, head_dim=256)),
        cache_config=NS(cache_dtype="auto", mamba_ssm_cache_dtype="float32"),
        offload_config=NS(uva=NS(cpu_offload_gb=0), prefetch=NS(offload_group_size=0)),
    )
    worker = NS(vllm_config=cfg)
    validate_worker_config(worker)
    for section, field in (("uva", "cpu_offload_gb"), ("prefetch", "offload_group_size")):
        setattr(getattr(cfg.offload_config, section), field, 1)
        with pytest.raises(RuntimeError, match="offload"):
            validate_worker_config(worker)
        setattr(getattr(cfg.offload_config, section), field, 0)


def test_gdn_backend_must_be_explicit_in_worker_contract():
    from vllm_mach.fp8.worker import validate_worker_config
    cfg = NS(
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           decode_context_parallel_size=1, enable_dbo=False),
        speculative_config=None,
        additional_config={'gdn_prefill_backend': 'flashinfer'},
        model_config=NS(dtype='torch.bfloat16', quantization='compressed-tensors',
            hf_text_config=NS(model_type='qwen3_5_text', hidden_size=2560,
                intermediate_size=9216, num_hidden_layers=32, num_attention_heads=16,
                num_key_value_heads=4, head_dim=256)),
        cache_config=NS(cache_dtype='auto', mamba_ssm_cache_dtype='float32'))
    with pytest.raises(RuntimeError, match='gdn_prefill_backend=triton'):
        validate_worker_config(NS(vllm_config=cfg))


def test_fla_choices_require_exact_scalar_dtype_key_and_available_config():
    import json
    import sys
    from vllm_mach.fp8 import prefill
    rows = json.loads((Path(prefill.__file__).parent / 'data/fla_choices.json').read_text())['kernels']
    class Config:
        def __init__(self, **values): self.values = values
        def __eq__(self, other): return isinstance(other, Config) and self.values == other.values
    with patch.dict(sys.modules, {'triton': NS(Config=Config)}), patch.object(prefill, '_UNKNOWN', {}), patch.object(prefill, '_QUALIFIED', {}):
        for row in rows:
            n = next(i for i, v in enumerate(row['key']) if isinstance(v, str))
            keys = ['key' + str(i) for i in range(n)]
            values = dict(zip(keys, row['key'][:n]))
            values.update({str(i): NS(dtype=v) for i, v in enumerate(row['key'][n:])})
            wanted = Config(**row['selected'])
            tuner = NS(keys=keys, arg_names=list(values), configs=[wanted], cache={})
            assert prefill.choose(tuner, row, values) == row['key']
            assert tuner.cache[tuple(row['key'])] == wanted
            changed = {**values, keys[0]: values[keys[0]] + 1}
            assert prefill.choose(tuner, row, changed) is None
            changed = dict(values)
            changed['0'] = NS(dtype='torch.float32' if row['key'][n] != 'torch.float32' else 'torch.bfloat16')
            assert prefill.choose(tuner, row, changed) is None
            tuner.configs = []
            with pytest.raises(RuntimeError, match='Config disappeared'):
                prefill.choose(tuner, row, values)
            tuner.configs = [wanted]
            tuner.cache[tuple(row['key'])] = Config(kwargs={'bad': 1})
            with pytest.raises(RuntimeError, match='conflicting tuning'):
                prefill.choose(tuner, row, values)


def test_legacy_installers_leave_unselected_profiles_untouched():
    from vllm_mach.fp8 import prefill, guard_norm, quantize
    with patch.dict(os.environ, {}, clear=True), patch.object(quantize, '_INSTALLED', False), patch.object(prefill, '_INSTALLED', False), patch.object(guard_norm, '_STATE', None):
        assert prefill.install() is False
        assert guard_norm.install() is False
        assert quantize.install() is False


def test_exact_division_selector_keeps_unknown_formats_on_original_op():
    import sys
    from vllm_mach.fp8 import quantize
    dtype = NS(bfloat16='bf16', float8_e4m3fn='fp8', float32='fp32')
    device = object()
    x = NS(ndim=2, shape=(16, 2560), dtype='bf16', is_cuda=True,
           device=device, is_contiguous=lambda: True)
    q = NS(ndim=2, shape=x.shape, dtype='fp8', device=device, is_contiguous=lambda: True)
    s = NS(ndim=2, shape=(16, 20), dtype='fp32', device=device, stride=lambda: (1, 16))
    args = [128, 1e-10, -448., 448., False, True, False]
    with patch.dict(sys.modules, {'torch': dtype}):
        assert quantize.eligible(x, q, s, *args)
        for index, invalid in [(0, 64), (1, 1e-6), (4, True), (5, False), (6, True)]:
            changed = list(args)
            changed[index] = invalid
            assert not quantize.eligible(x, q, s, *changed)
        with patch.object(s, 'stride', return_value=(20, 1)):
            assert not quantize.eligible(x, q, s, *args)
        with patch.object(x, 'dtype', 'fp32'):
            assert not quantize.eligible(x, q, s, *args)
        with patch.object(x, 'is_cuda', False):
            assert not quantize.eligible(x, q, s, *args)


def test_fla_qualification_is_reused_but_mutations_and_cache_conflicts_still_fail():
    import copy
    import sys
    from vllm_mach.fp8 import prefill

    class Config:
        constructions = 0

        def __init__(self, **values):
            Config.constructions += 1
            self.values = values

        def __eq__(self, other):
            return isinstance(other, Config) and self.values == other.values

    class Configs(list):
        scans = 0

        def __contains__(self, value):
            self.scans += 1
            return super().__contains__(value)

    row = dict(function='kernel', key=[32, 'torch.bfloat16'],
               selected=dict(kwargs={'BK': 64}, num_warps=4, num_stages=3))
    values = dict(N=32, x=NS(dtype='torch.bfloat16'))
    member = Config(**copy.deepcopy(row['selected']))
    configs = Configs([member])
    tuner = NS(keys=['N'], arg_names=['N', 'x'], configs=configs, cache={})
    with patch.dict(sys.modules, {'triton': NS(Config=Config)}), \
            patch.object(prefill, '_QUALIFIED', {}), patch.object(prefill, '_UNKNOWN', {}):
        # Installation can qualify before the first actual invocation.
        state = prefill._qualify(tuner, row)
        constructions, scans = Config.constructions, configs.scans
        for _ in range(3):
            assert prefill.choose(tuner, row, values) == row['key']
            assert tuner.cache[tuple(row['key'])] is state['config']
        assert (Config.constructions, configs.scans) == (constructions, scans)
        assert prefill.choose(tuner, row, dict(values, N=33)) is None
        assert tuner.cache.keys() == {tuple(row['key'])}
        # Equal external entries retain the old acceptance semantics.
        tuner.cache[tuple(row['key'])] = Config(**copy.deepcopy(row['selected']))
        assert prefill.choose(tuner, row, values) == row['key']
        for bad in (None, Config(kwargs={'BK': 128})):
            tuner.cache[tuple(row['key'])] = bad
            with pytest.raises(RuntimeError, match='conflicting tuning'):
                prefill.choose(tuner, row, values)
        tuner.cache.clear()
        tuner.configs = Configs([Config(kwargs={'BK': 128}), member])
        assert prefill.choose(tuner, row, values) == row['key']
        assert state['selected_index'] == 1
        member.values['kwargs']['BK'] = 128
        with pytest.raises(RuntimeError, match='Config disappeared'):
            prefill.choose(tuner, row, values)
        member.values['kwargs']['BK'] = 64
        assert prefill.choose(tuner, row, values) == row['key']
        state['config'].values['kwargs']['BK'] = 128
        with pytest.raises(RuntimeError, match='Config mutated'):
            prefill.choose(tuner, row, values)
        state['config'].values['kwargs']['BK'] = 64
        changed = copy.deepcopy(row)
        changed['selected']['num_warps'] = 8
        with pytest.raises(RuntimeError, match='policy changed'):
            prefill.choose(tuner, changed, values)
