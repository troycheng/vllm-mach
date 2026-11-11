"""Dense FLA backend and exact compile bindings; no Torch/CUDA dependency."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

from vllm_mach.mxfp6.gdn import dense_prefill, dense_prep, legacy
from vllm_mach.runtime import attention


def config():
    weight = dict(dtype='fp6_e3m2', group_size=32, qscheme='per_group',
                  scale_format='e8m0', is_dynamic=False, symmetric=True)
    activation = dict(dtype='fp8_e4m3', group_size=32, qscheme='per_group',
                      scale_format='e8m0', is_dynamic=True, symmetric=True)
    text = N(model_type='qwen3_5_text', hidden_size=5120, num_hidden_layers=64,
             linear_num_key_heads=16, linear_num_value_heads=48,
             linear_key_head_dim=128, linear_value_head_dim=128)
    return N(model_config=N(dtype='torch.bfloat16', quantization='quark', hf_text_config=text,
                           model_arch_config=N(quantization_config={'global_quant_config':
                           {'weight': weight, 'input_tensors': activation}})),
             parallel_config=N(tensor_parallel_size=2, pipeline_parallel_size=1,
                               decode_context_parallel_size=1, enable_dbo=False),
             cache_config=N(cache_dtype='auto', mamba_ssm_cache_dtype='float32'),
             speculative_config=None, additional_config={})


class DensePrefillAdmission(unittest.TestCase):
    def setUp(self):
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        for context in [patch.dict(os.environ, {}, clear=True),
                        patch.object(attention, '_NATIVE_REGISTERED', True),
                        patch.object(dense_prefill, '_ENABLED', False),
                        patch.object(dense_prefill, '_INSTALLED', False),
                        patch.object(dense_prep, '_sm120', return_value=True)]:
            self.stack.enter_context(context)

    def test_real_legacy_entry_selects_dense_only_without_moe_bundle(self):
        for dtype in ('float32', 'float16'):
            cfg = config()
            cfg.cache_config.mamba_ssm_cache_dtype = dtype
            with self.subTest(dtype=dtype), patch.object(dense_prep, 'install') as prep, patch.object(
                    dense_prefill, 'install') as prefill, patch.object(legacy, 'install') as moe:
                self.assertEqual(legacy.resolve_backend(cfg, 'auto'), 'triton')
                self.assertTrue(dense_prefill._ENABLED)
                self.assertFalse(legacy.enabled())
                prep.assert_called_once()
                prefill.assert_called_once()
                moe.assert_not_called()

    def test_preserves_every_explicit_backend_including_auto(self):
        for requested in ('auto', 'flashinfer', 'cutedsl', 'triton', 'unknown'):
            cfg = config()
            cfg.additional_config = {'gdn_prefill_backend': requested}
            with self.subTest(requested=requested), patch.object(dense_prefill, 'install') as install:
                self.assertEqual(dense_prefill.resolve_backend(cfg, requested), requested)
                self.assertEqual(dense_prefill._ENABLED, requested == 'triton')
                self.assertEqual(install.call_count, int(requested == 'triton'))

    def test_other_state_model_geometry_and_hardware_remain_upstream(self):
        cases = [('cache_config.mamba_ssm_cache_dtype', 'bfloat16'),
                 ('model_config.hf_text_config.model_type', 'qwen3_5_moe_text'),
                 ('model_config.hf_text_config.linear_num_value_heads', 32),
                 ('model_config.quantization', 'fp8'),
                 ('parallel_config.tensor_parallel_size', 1),
                 ('speculative_config', object())]
        for path, value in cases:
            cfg = copy.deepcopy(config()); obj = cfg; names = path.split('.')
            for part in names[:-1]: obj = getattr(obj, part)
            setattr(obj, names[-1], value)
            with self.subTest(path=path), patch.object(dense_prefill, 'install') as install:
                self.assertEqual(dense_prefill.resolve_backend(cfg, 'auto'), 'auto')
                self.assertFalse(dense_prefill._ENABLED)
                install.assert_not_called()
        for context in [patch.object(dense_prep, '_sm120', return_value=False),
                        patch.object(attention, '_NATIVE_REGISTERED', False),
                        patch.dict(os.environ, {'VLLM_MACH_PROFILE': 'another-profile'})]:
            with context, patch.object(dense_prefill, 'install') as install:
                self.assertEqual(dense_prefill.resolve_backend(config(), 'auto'), 'auto')
                install.assert_not_called()


class DensePrefillBindings(unittest.TestCase):
    def test_packaged_six_functions_and_two_state_dtypes_are_shared_not_rank_scoped(self):
        rows = json.loads((dense_prefill._DATA / 'dense_fla_choices.json').read_text())['kernels']
        self.assertEqual(len(rows), 7)
        self.assertEqual(len({r['function'] for r in rows}), 6)
        self.assertTrue(all('rank' not in r for r in rows))
        state_rows = [r for r in rows if r['function'] == 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64']
        self.assertEqual({r['key'][-4] for r in state_rows}, {'torch.float32', 'torch.float16'})
        self.assertEqual(state_rows[0]['selected'], state_rows[1]['selected'])
        self.assertEqual(state_rows[0]['key'][:-4] + state_rows[0]['key'][-3:],
                         state_rows[1]['key'][:-4] + state_rows[1]['key'][-3:])
        self.assertTrue(all(24 in r['key'] for r in rows))

    def test_cached_run_forwarding_unknown_key_drift_and_completed_counters(self):
        rows = json.loads((dense_prefill._DATA / 'dense_fla_choices.json').read_text())['kernels']
        class Config:
            def __init__(self, **values): self.values = values
            def __eq__(self, other): return isinstance(other, Config) and self.values == other.values
        class Autotuner:
            def __init__(self, row):
                self.keys = [f's{i}' for i, v in enumerate(row['key']) if not isinstance(v, str)]
                self.arg_names = self.keys + [f't{i}' for i, v in enumerate(row['key']) if isinstance(v, str)]
                self.configs = [Config(**row['selected'])]; self.cache = {}; self.seen = []
            def run(self, *args, **kwargs):
                self.seen.append((args, kwargs, dict(self.cache)))
                if kwargs.get('fail'): raise RuntimeError('original failed')
                return N(hash='actual-result-hash')
        modules = {}; tuners = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.py'; source.write_text('# source identity\n')
            gates = {'chunk_o.py': hashlib.sha256(source.read_bytes()).hexdigest()}
            (root / 'legacy_sources.json').write_text(json.dumps(gates))
            (root / 'dense_fla_choices.json').write_text(json.dumps({'kernels': rows}))
            for row in rows:
                module = modules.setdefault(row['module'], N(__file__=str(source)))
                tuner = Autotuner(row); setattr(module, row['function'], tuner); tuners.append(tuner)
            fake = {'triton': N(Config=Config), 'triton.runtime.autotuner': N(Autotuner=Autotuner)}
            with patch.dict('sys.modules', fake), patch.object(dense_prefill, '_DATA', root), patch.object(
                    dense_prefill.importlib, 'import_module', side_effect=modules.__getitem__), patch.object(
                    dense_prefill, '_ENABLED', True), patch.object(dense_prefill, '_INSTALLED', False), patch.object(
                    dense_prefill, '_STATS', {'calls': {}, 'choices': {}}):
                dense_prefill.install(); bound = tuners[0].run; dense_prefill.install()
                self.assertIs(bound, tuners[0].run)
                state_rows = [r for r in rows if r['function'] == 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64']
                state_tuner = getattr(modules[state_rows[0]['module']], state_rows[0]['function'])
                self.assertIs(state_tuner.run.__wrapped__.__func__, Autotuner.run)
                for binding in state_rows:
                    values = [v for v in binding['key'] if not isinstance(v, str)]
                    values += [N(dtype=v) for v in binding['key'] if isinstance(v, str)]
                    state_tuner.run(*values)
                    self.assertIn(tuple(binding['key']), state_tuner.cache)
                self.assertEqual(dense_prefill._STATS['calls'][state_rows[0]['function']], 2)
                row = rows[0]; tuner = tuners[0]
                args = [v for v in row['key'] if not isinstance(v, str)]
                args += [N(dtype=v) for v in row['key'] if isinstance(v, str)]
                self.assertEqual(tuner.run(*args).hash, 'actual-result-hash')
                self.assertEqual(dense_prefill._STATS['calls'][row['function']], 1)
                self.assertEqual(tuner.cache[tuple(row['key'])], Config(**row['selected']))
                self.assertEqual(tuner.seen[-1][:2], (tuple(args), {}))
                count = len(tuner.cache)
                tuner.run(**dict(zip(tuner.arg_names, [25, *args[1:]])))
                self.assertEqual(len(tuner.cache), count)
                self.assertEqual(dense_prefill._STATS['calls'][row['function']], 1)
                with self.assertRaisesRegex(RuntimeError, 'original failed'): tuner.run(*args, fail=True)
                self.assertEqual(dense_prefill._STATS['calls'][row['function']], 1)
                tuner.cache[tuple(row['key'])] = Config(kwargs={}, num_warps=1, num_stages=1, num_ctas=1)
                with self.assertRaisesRegex(RuntimeError, 'selection changed'): tuner.run(*args)
                tuner.cache.clear(); tuner.configs.clear()
                with self.assertRaisesRegex(RuntimeError, 'selection changed'): tuner.run(*args)
                dense_prefill._ENABLED = False
                tuner.run(*args)
                self.assertEqual(dense_prefill._STATS['calls'][row['function']], 1)
                self.assertEqual(dense_prefill.receipt()['choices'][row['function']]['hash'], 'actual-result-hash')
                dense_prefill._INSTALLED = False; source.write_text('# foreign source\n')
                with self.assertRaisesRegex(RuntimeError, 'source changed'): dense_prefill.install()


if __name__ == '__main__': unittest.main()
