"""Native FP16 restoration admission; no CUDA/Torch import is required."""
import copy
import os
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch
from vllm_mach.mxfp6.gdn import legacy
from vllm_mach.runtime import attention


def config():
    weight = dict(dtype='fp6_e3m2', group_size=32, qscheme='per_group',
                  scale_format='e8m0', is_dynamic=False, symmetric=True)
    activation = dict(dtype='fp8_e4m3', group_size=32, qscheme='per_group',
                      scale_format='e8m0', is_dynamic=True, symmetric=True)
    text = N(model_type='qwen3_5_moe_text', hidden_size=2048, num_hidden_layers=40,
             linear_num_key_heads=16, linear_num_value_heads=32,
             linear_key_head_dim=128, linear_value_head_dim=128,
             num_attention_heads=16, num_key_value_heads=2, head_dim=256)
    return N(model_config=N(dtype='torch.bfloat16', quantization='quark', hf_text_config=text,
                           model_arch_config=N(quantization_config={'global_quant_config':
                           {'weight': weight, 'input_tensors': activation}})),
             parallel_config=N(tensor_parallel_size=2, pipeline_parallel_size=1,
                               decode_context_parallel_size=1, enable_dbo=False),
             cache_config=N(cache_dtype='auto', mamba_ssm_cache_dtype='float16'),
             speculative_config=None, additional_config={})


class NativeLegacyAdmission(unittest.TestCase):
    def setUp(self):
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'VLLM_QWEN3_5_FP16_SSM': '1'}, clear=True))
        self.stack.enter_context(patch.object(attention, '_NATIVE_REGISTERED', True))
        self.stack.enter_context(patch.object(legacy, '_ENABLED', False))
        self.stack.enter_context(patch.object(legacy, '_sm120', return_value=True))

    def test_default_and_explicit_backend_semantics(self):
        cfg = config()
        with patch.object(legacy, '_sm120', return_value=False), patch.object(legacy, 'install') as inactive:
            self.assertEqual(legacy.resolve_backend(cfg, 'auto'), 'auto')
            inactive.assert_not_called()
        with patch.object(legacy, 'install') as install:
            self.assertEqual(legacy.resolve_backend(cfg, 'auto'), 'triton')
            install.assert_called_once()
        for value in ('auto', 'flashinfer', 'cutedsl', 'unknown'):
            cfg.additional_config = {'gdn_prefill_backend': value}
            with patch.object(legacy, 'install') as install:
                self.assertEqual(legacy.resolve_backend(cfg, value), value)
                self.assertFalse(legacy.enabled())
                install.assert_not_called()
        cfg.additional_config = {'gdn_prefill_backend': 'triton'}
        with patch.object(legacy, 'install') as install:
            self.assertEqual(legacy.resolve_backend(cfg, 'triton'), 'triton')
            install.assert_called_once()

    def test_default_fp32_dense_and_foreign_geometry_remain_upstream(self):
        for section, field, value in [
            ('cache_config', 'mamba_ssm_cache_dtype', 'float32'),
            ('model_config.hf_text_config', 'model_type', 'qwen3_5_text'),
            ('model_config.hf_text_config', 'linear_num_value_heads', 48),
            ('model_config', 'quantization', 'compressed-tensors'),
            ('parallel_config', 'tensor_parallel_size', 1),
            ('parallel_config', 'enable_dbo', True),
        ]:
            cfg = config(); obj = cfg
            for part in section.split('.'):
                obj = getattr(obj, part)
            setattr(obj, field, value)
            with self.subTest(field=field, value=value), patch.object(legacy, 'install') as install:
                self.assertEqual(legacy.resolve_backend(cfg, 'auto'), 'auto')
                self.assertFalse(legacy.enabled())
                install.assert_not_called()
        with patch.dict(os.environ, {'VLLM_QWEN3_5_FP16_SSM': '0'}):
            self.assertFalse(legacy.model_eligible(config()))
        cfg = config(); cfg.speculative_config = object()
        self.assertFalse(legacy.model_eligible(cfg))
        with patch.dict(os.environ, {'VLLM_MACH_PROFILE': 'qwen35-2b-mxfp8-champion-v1'}):
            self.assertFalse(legacy.model_eligible(config()))

    def test_attention_requires_actual_native_instance_context(self):
        stock = object()
        kw = dict(num_seqs=32, heads=8, kv_heads=1, segments=16,
                  head_size=256, padded_head_size=256, bf16=True, use_fp8=False)
        legacy._ENABLED = True
        with patch.object(legacy._CONTEXT, 'attention', None, create=True):
            self.assertIs(legacy.attention(stock, **kw), stock)
        with patch.object(legacy._CONTEXT, 'attention', 'model.layers.3.self_attn.attn', create=True):
            for key, value in [('num_seqs', 48), ('heads', 16), ('kv_heads', 2),
                               ('segments', 64), ('head_size', 128), ('bf16', False),
                               ('use_fp8', True)]:
                self.assertIs(legacy.attention(stock, **{**kw, key: value}), stock)

    def test_qualified_grid_rows_use_same_attention_kernel(self):
        stock = object()
        target = object()
        module = 'vllm_mach.mxfp8.two_b._attention_reduction_kernel'
        with patch.object(legacy, '_ENABLED', True), patch.object(
                legacy._CONTEXT, 'attention', 'model.layers.3.self_attn.attn', create=True), patch.dict(
                    'sys.modules', {module: N(reduce_segments_explicit_old_lane=target)}):
            for rows in (1, 2, 4, 8, 16, 24, 32):
                with self.subTest(rows=rows):
                    result = legacy.attention(stock, num_seqs=rows, heads=8, kv_heads=1,
                        segments=16, head_size=256, padded_head_size=256, bf16=True, use_fp8=False)
                    self.assertIsNot(result, stock)
        self.assertEqual(legacy._PACKED_ROWS, (16, 24, 32))
        self.assertEqual(legacy._DECODE_ROWS, (1, 2, 4, 8, 16, 24, 32))

    def test_prefill_wrapper_accepts_real_qwen_keyword_and_mixed_calls(self):
        try:
            import ast
            import functools
            import inspect
            from pathlib import Path
            import torch
            import triton
            from torch._subclasses.fake_tensor import FakeTensorMode
            from vllm.third_party.flash_linear_attention.ops import fused_gdn_prefill_post_conv as upstream
            from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
            from vllm_mach import _gdn_prep_kernel as shared
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        # Compile the actual nested product wrapper, retaining its real Python
        # argument binder. wraps() otherwise makes inspect.signature hide drift.
        tree = ast.parse(inspect.getsource(legacy.install))
        wrapper = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'prep')
        namespace = dict(vars(legacy), original_prep=upstream.fused_post_conv_prep,
                         torch=torch, _ENABLED=True, _STATS={'prep_calls': 0})
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<actual-native-prep-wrapper>', 'exec'), namespace)
        prep = namespace['prep']
        abi = lambda fn: [(n, p.kind, p.default) for n, p in inspect.signature(fn, follow_wrapped=False).parameters.items()]
        self.assertEqual(abi(prep), abi(upstream.fused_post_conv_prep))
        self.assertEqual(abi(prep), abi(shared.fused_post_conv_prep))
        prepare_tree = ast.parse(inspect.getsource(legacy.prepare))
        decode = next(n for n in prepare_tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'decode_norm')
        self.assertEqual([p.arg for p in decode.args.args], list(inspect.signature(qwen.QwenGatedDeltaNetAttention._rms_norm_gated_cuda).parameters)[1:])
        sentinel = object()
        fallback = dict(vars(legacy), torch=torch, layer=N(get_state_dtype=lambda: (torch.bfloat16, torch.float32)), original_decode=lambda x, output_gate, out: sentinel)
        exec(compile(ast.Module(body=[decode], type_ignores=[]), '<actual-native-decode-norm-wrapper>', 'exec'), fallback)
        self.assertIs(fallback['decode_norm'](x=object(), output_gate=object(), out=object()), sentinel)
        calls = []
        class DeviceLaunch:
            def __init__(self, name): self.name = name
            def __getitem__(self, grid):
                def run(*args, **kwargs): calls.append((self.name, grid, args, kwargs))
                return run
        with FakeTensorMode(), patch.object(upstream, 'triton', triton), patch.object(shared, 'triton', triton), patch.object(upstream, '_fused_post_conv_kernel', DeviceLaunch('upstream')), patch.object(shared, '_fused_post_conv_kernel', DeviceLaunch('qualified')):
            conv = torch.empty(64, 4096, dtype=torch.bfloat16, device='cuda:0')
            a = torch.empty(64, 16, dtype=torch.bfloat16, device='cuda:0')
            values = dict(conv_output=conv, a=a, b=a, A_log=torch.empty(16, device='cuda:0'), dt_bias=torch.empty(16, dtype=torch.bfloat16, device='cuda:0'), num_k_heads=8, head_k_dim=128, head_v_dim=128, apply_l2norm=True, output_g_exp=False)
            # Both real qwen warmup and runtime callsites are keyword calls.
            sites = [n for n in ast.walk(ast.parse(Path(qwen.__file__).read_text())) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fused_post_conv_prep']
            self.assertEqual(len(sites), 2)
            for call in sites:
                self.assertFalse(call.args)
                self.assertEqual({k.arg for k in call.keywords}, set(values))
                self.assertEqual(len(prep(**{k.arg: values[k.arg] for k in call.keywords})), 5)
            self.assertEqual(len(prep(*values.values())), 5)
            first = list(values)[:5]
            self.assertEqual(len(prep(*(values[k] for k in first), **{k: v for k, v in values.items() if k not in first})), 5)
            # An unmatched frontend keeps the actual upstream body/allocation.
            self.assertEqual(len(prep(**{**values, 'apply_l2norm': False})), 5)
        self.assertEqual([x[0] for x in calls], ['qualified'] * 4 + ['upstream'])
        self.assertEqual(namespace['_STATS']['prep_calls'], 4)
        self.assertFalse(torch.cuda.is_initialized())




class NativeQualificationCache(unittest.TestCase):
    class Config:
        def __init__(self, kwargs=None, num_warps=4, num_ctas=1, num_stages=3,
                     maxnreg=None, pre_hook=None, ir_override=None):
            self.kwargs = dict(kwargs or {})
            self.num_warps, self.num_ctas, self.num_stages = num_warps, num_ctas, num_stages
            self.maxnreg, self.pre_hook, self.ir_override = maxnreg, pre_hook, ir_override

        def __eq__(self, other):
            return isinstance(other, type(self)) and legacy._config_state(self) == legacy._config_state(other)

    def fixture(self):
        row = {'selected': dict(kwargs={'BK': 64}, num_warps=4, num_stages=3)}
        tuner = N(configs=[self.Config(num_warps=2), self.Config(**row['selected'])], cache={})
        return row, tuner, legacy._QualifiedChoice(tuner, row, self.Config)

    def test_same_choice_equivalent_external_cache_and_missing_cache(self):
        row, tuner, qualified = self.fixture()
        key = (16, 128, 'torch.bfloat16')
        choice = qualified.choose(tuner, row, key)
        tuner.cache[key] = choice
        self.assertIs(qualified.choose(tuner, row, key), choice)
        tuner.cache[key] = self.Config(**row['selected'])
        self.assertIs(qualified.choose(tuner, row, key), choice)
        tuner.cache.clear()
        self.assertIs(qualified.choose(tuner, row, key), choice)

    def test_mutations_preserve_original_membership_and_cache_contract(self):
        for case in ('cache', 'member', 'removed', 'choice', 'row', 'reordered'):
            row, tuner, qualified = self.fixture()
            key = ('key',)
            if case == 'cache': tuner.cache[key] = self.Config(num_warps=8)
            elif case == 'member': tuner.configs[1].kwargs['BK'] = 32
            elif case == 'removed': tuner.configs.pop()
            elif case == 'choice': qualified.choice.num_stages = 5
            elif case == 'row': row['selected']['kwargs']['BK'] = 32
            elif case == 'reordered': tuner.configs.reverse()
            expected = self.Config(**row['selected'])
            reject = expected not in tuner.configs or (key in tuner.cache and tuner.cache[key] != expected)
            with self.subTest(case=case):
                if reject:
                    with self.assertRaisesRegex(RuntimeError, 'selection changed'):
                        qualified.choose(tuner, row, key)
                else:
                    self.assertEqual(qualified.choose(tuner, row, key), expected)

    def test_compiled_metadata_and_ptx_cache_invalidates_on_real_changes(self):
        result = N(metadata=N(hash='h', num_warps=1, num_stages=3), asm={'ptx': 'actual PTX'})
        cache = {}
        original = legacy.hashlib.sha256
        with patch.object(legacy.hashlib, 'sha256', wraps=original) as digest:
            first = legacy._compiled_info(result, cache)
            self.assertIs(legacy._compiled_info(result, cache), first)
            self.assertEqual(digest.call_count, 1)
            result.asm['ptx'] = 'changed actual PTX'
            changed = legacy._compiled_info(result, cache)
            self.assertNotEqual(changed['ptx_sha256'], first['ptx_sha256'])
            result.metadata.num_warps = 4
            self.assertEqual(legacy._compiled_info(result, cache)['num_warps'], 4)
            other = N(metadata=result.metadata, asm=result.asm)
            legacy._compiled_info(other, cache)
            self.assertEqual(digest.call_count, 4)

    def test_actual_tuner_wrapper_disabled_unknown_and_receipt(self):
        import ast
        import functools
        import inspect
        row, tuner, qualified = self.fixture()
        row.update(key=[16], function='actual_fixture')
        tuner.arg_names, tuner.keys = ['H'], ['H']
        calls = []
        result = N(hash='actual-result')
        original = lambda *a, **kw: calls.append((a, kw)) or result
        tree = ast.parse(inspect.getsource(legacy.install))
        run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run')
        namespace = dict(functools=functools, original=original, row=row,
                         qualified=qualified, _ENABLED=False, _STATS={'pins': {}, 'pin_choices': {}})
        exec(compile(ast.Module(body=[run], type_ignores=[]), '<actual-native-tuner-run>', 'exec'), namespace)
        wrapper = namespace['run']
        self.assertIs(wrapper(tuner, 16, grid=(1,)), result)
        self.assertFalse(tuner.cache)
        namespace['_ENABLED'] = True
        self.assertIs(wrapper(tuner, 32, grid=(1,)), result)
        self.assertFalse(tuner.cache)
        wrapper(tuner, 16, grid=(1,))
        self.assertIs(tuner.cache[(16,)], qualified.choice)
        self.assertEqual(namespace['_STATS']['pins'], {'actual_fixture': 1})
        self.assertEqual(namespace['_STATS']['pin_choices']['actual_fixture']['hash'], 'actual-result')
        self.assertEqual(calls, [((16,), {'grid': (1,)}), ((32,), {'grid': (1,)}), ((16,), {'grid': (1,)})])

    def test_actual_norm_wrapper_keeps_dynamic_capture_and_receipt(self):
        import ast
        import inspect
        result = N(metadata=N(hash='compiled', num_warps=1, num_stages=3), asm={'ptx': 'norm PTX'})
        calls = []
        target = N(run=lambda *a, **kw: calls.append((a, kw)) or result)
        capture = unittest.mock.Mock(side_effect=[False, True])
        tree = ast.parse(inspect.getsource(legacy.install))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'norm_run')
        stats = dict(norm_calls={}, norm_capture={}, norm_choices={})
        original = unittest.mock.Mock(return_value='upstream')
        namespace = dict(vars(legacy), _ENABLED=True, _STATS=stats, norm_compiled={},
                         original_norm=original, torch=N(cuda=N(is_current_stream_capturing=capture)))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-native-norm-run>', 'exec'), namespace)
        args = tuple([None] * 10 + [2167, 128, 1e-6])
        kw = dict(BLOCK_N=128, HAS_BIAS=False, HAS_Z=True, IS_RMS_NORM=True,
                  NORM_BEFORE_GATE=True, num_warps=1, ROWS_PER_BLOCK=4, ACTIVATION='silu')
        with patch.object(legacy._CONTEXT, 'norm', 'layer0', create=True), patch.dict(
                'sys.modules', {'vllm_mach.fp8._guard_kernel': N(layer_norm_fwd_kernel=target)}):
            for _ in range(2): self.assertIs(namespace['norm_run'](*args, **kw), result)
            namespace['_ENABLED'] = False
            self.assertEqual(namespace['norm_run'](*args, **kw), 'upstream')
        self.assertEqual(calls, [(args, kw), (args, kw)])
        self.assertEqual(capture.call_count, 2)
        self.assertEqual(stats['norm_calls'], {'layer0:row4': 2})
        self.assertEqual(stats['norm_capture'], {'layer0:row4:m2167': 1})
        self.assertTrue(stats['norm_choices']['layer0:row4']['capturing'])
        self.assertEqual(stats['norm_choices']['layer0:row4']['hash'], 'compiled')


class NativeGuardedTunerCache(unittest.TestCase):
    Config = NativeQualificationCache.Config

    def fixture(self, *, single=False, leaf_error=False):
        import ast
        import functools
        import inspect
        from types import MethodType
        from vllm_mach.mxfp6.gdn.qualified_cache import attach
        row, tuner, _ = NativeQualificationCache.fixture(self)
        row.update(key=[16], function='actual_fixture')
        if single:
            tuner.configs = [self.Config(**row['selected'])]
        qualified = legacy._QualifiedChoice(tuner, row, self.Config)
        tuner.nargs = None
        calls = []
        def original(h, **kwargs):
            tuner.nargs = {'H': h}
            config = tuner.cache[(h,)] if len(tuner.configs) > 1 and (h,) in tuner.cache else tuner.configs[0]
            calls.append((h, kwargs, config))
            if leaf_error:
                raise ValueError('leaf sentinel')
            tuner.nargs = None
            return config
        tuner.arg_names, tuner.keys = ['H'], ['H']
        tree = ast.parse(inspect.getsource(legacy.install))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run')
        namespace = dict(functools=functools, original=original, row=row,
                         qualified=qualified, _ENABLED=True,
                         _STATS={'pins': {}, 'pin_choices': {}})
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-rare-tuner-run>', 'exec'), namespace)
        baseline = MethodType(namespace['run'], tuner)
        raw = tuner.cache
        run, cache = attach(tuner, row, qualified, lambda: namespace['_ENABLED'], original, baseline)
        self.assertIs(cache.raw, raw)
        return row, tuner, qualified, namespace, run, cache, calls

    def test_qualified_source_is_same_as_actual_cpu_and_gpu_target(self):
        import ast
        import hashlib
        import inspect
        import json
        from vllm_mach.mxfp6.gdn import qualified_cache
        def canonical(value):
            if isinstance(value, ast.AST):
                return [type(value).__name__, [(k, canonical(v)) for k, v in ast.iter_fields(value)
                        if not (k == 'type_params' and v == [])]]
            if isinstance(value, list):
                return [canonical(v) for v in value]
            return value
        tree = ast.parse(inspect.getsource(qualified_cache))
        tree.body = tree.body[1:]  # Only module description differs from work fixture8e45.
        digest = hashlib.sha256(json.dumps(canonical(tree), separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, '412499674e8fcfb3dcde7ec82d00a26fccd22f3ebdb434f1c57b76b533cb57ed')
        install = ast.parse(inspect.getsource(legacy.install))
        binding = next(n for n in ast.walk(install) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == 'attach')
        self.assertEqual([ast.unparse(n) for n in binding.args],
                         ['tuner', 'row', 'qualified', 'enabled', 'original', 'baseline'])

    def test_raw_alias_unknown_disabled_clear_and_matching_lookup(self):
        row, tuner, qualified, ns, run, cache, calls = self.fixture()
        external = self.Config(num_warps=2)
        cache.raw[(32,)] = external
        self.assertIs(run(32), external)
        self.assertIsNone(cache.first_real_lookup)
        cache.raw[(16,)] = external
        ns['_ENABLED'] = False
        self.assertIs(run(16), external)
        self.assertIs(cache.raw[(16,)], external)
        ns['_ENABLED'] = True
        cache.clear()
        self.assertFalse(cache.raw)
        with patch.object(qualified, 'choose', wraps=qualified.choose) as choose:
            self.assertIs(run(16, grid=(1,)), qualified.choice)
            self.assertEqual(choose.call_count, 1)
        self.assertEqual(cache.first_real_lookup, (16,))
        self.assertEqual(calls[-1][1], {'grid': (1,)})
        cache.clear()
        self.assertIs(run(16), qualified.choice)
        self.assertIs(tuner.cache.raw, cache.raw)

    def test_qualification_rejection_restores_entry_nargs_but_leaf_error_does_not(self):
        _, tuner, _, _, run, cache, calls = self.fixture()
        prior = {'entry': 'unchanged'}
        tuner.nargs = prior
        cache.raw[(16,)] = self.Config(num_warps=32)
        with self.assertRaisesRegex(RuntimeError, 'selection changed'):
            run(16)
        self.assertIs(tuner.nargs, prior)
        self.assertFalse(calls)
        self.assertIsNone(cache.qualification_error)
        _, tuner, _, _, run, cache, _ = self.fixture(leaf_error=True)
        with self.assertRaisesRegex(ValueError, 'leaf sentinel'):
            run(16)
        self.assertEqual(tuner.nargs, {'H': 16})
        self.assertIsNone(cache.qualification_error)

    def test_single_config_fallback_selects_once_and_restores_bypass(self):
        _, tuner, qualified, _, run, cache, _ = self.fixture(single=True)
        with patch.object(qualified, 'choose', wraps=qualified.choose) as choose:
            self.assertEqual(run(16), qualified.choice)
            self.assertEqual(choose.call_count, 1)
        self.assertEqual(cache.raw_contains_depth, 0)
        cache.raw[(16,)] = self.Config(num_warps=32)
        with self.assertRaisesRegex(RuntimeError, 'selection changed'):
            run(16)
        self.assertEqual(cache.raw_contains_depth, 0)
        _, tuner, _, _, run, cache, _ = self.fixture(single=True, leaf_error=True)
        with self.assertRaisesRegex(ValueError, 'leaf sentinel'):
            run(16)
        self.assertEqual(cache.raw_contains_depth, 0)
        self.assertEqual(tuner.nargs, {'H': 16})


if __name__ == '__main__':
    unittest.main()
