"""Private projection lifecycle only; numerical/capture qualification uses CUDA."""
import contextlib
import types
import unittest
from unittest.mock import patch

from vllm_mach.mxfp6 import gdn
from vllm_mach.mxfp6.gdn import direct_norm, legacy


class DirectNormContracts(unittest.TestCase):
    def test_closure_values_preserve_cell_identity_and_empty_closures(self):
        first, second = object(), object()

        def with_cells():
            return first, second

        def without_cells():
            return None

        values = direct_norm.closure_values(with_cells)
        self.assertIs(values['first'], first)
        self.assertIs(values['second'], second)
        self.assertEqual(direct_norm.closure_values(without_cells), {})

    def projection(self, allowed, failure=None):
        events = []
        previous, result, z = object(), object(), object()
        native = types.SimpleNamespace(_CONTEXT=types.SimpleNamespace(norm=previous))
        layer = types.SimpleNamespace(prefix='model.layers.0.linear_attn')

        class Core:
            def flatten(core, dim):
                self.assertEqual(dim, -2)
                events.append('flatten')
                return core

        core = Core()

        def norm(x, gate, out):
            self.assertIs(x, core)
            self.assertIs(gate, z)
            self.assertIs(out, core)
            self.assertEqual(native._CONTEXT.norm, layer.prefix)
            events.append('direct_norm')
            if failure is not None:
                raise failure

        def original(x, gate):
            self.assertIs(x, core)
            self.assertIs(gate, z)
            events.append('complete_original_projection')
            return result

        def out_proj(x):
            self.assertIs(x, core)
            self.assertIs(native._CONTEXT.norm, previous)
            events.append('out_proj')
            return result, None

        layer._rms_norm_gated_cuda = norm
        layer.out_proj = out_proj
        projection = direct_norm._bind_projection(
            layer, original, object(), object(), native, object())
        with patch.object(direct_norm, 'eligible', return_value=allowed):
            if failure is None or not allowed:
                self.assertIs(projection(core, z), result)
            else:
                with self.assertRaises(type(failure)) as caught:
                    projection(core, z)
                self.assertIs(caught.exception, failure)
        self.assertIs(native._CONTEXT.norm, previous)
        return events

    def test_qualified_projection_normalizes_full_private_core_then_projects_once(self):
        self.assertEqual(self.projection(True), ['direct_norm', 'flatten', 'out_proj'])

    def test_rejected_projection_calls_complete_original_once(self):
        self.assertEqual(self.projection(False), ['complete_original_projection'])

    def test_norm_failure_restores_context_and_never_projects(self):
        self.assertEqual(self.projection(True, RuntimeError('norm failed')), ['direct_norm'])

    def test_prepare_binds_once_and_preserves_custom_projection(self):
        class Layer:
            def _output_projection(self, core, z):
                return core

        modules = {
            'torch': types.SimpleNamespace(),
            'vllm.model_executor.layers.layernorm': types.SimpleNamespace(RMSNormGated=object),
            'vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn':
                types.SimpleNamespace(QwenGatedDeltaNetAttention=Layer),
            'vllm.third_party.flash_linear_attention.ops':
                types.SimpleNamespace(layernorm_guard=object()),
        }
        layer = Layer()
        with patch.dict('sys.modules', modules), \
                patch.object(direct_norm, 'qualify_layer', return_value={}) as qualify:
            direct_norm.prepare(layer)
            installed = layer._output_projection
            self.assertIs(installed.__self__, layer)
            self.assertIsNot(installed.__func__, Layer._output_projection)
            direct_norm.prepare(layer)
            self.assertIs(layer._output_projection, installed)
            qualify.assert_called_once()
            custom = Layer()
            custom._output_projection = lambda core, z: core
            original = custom._output_projection
            direct_norm.prepare(custom)
            self.assertIs(custom._output_projection, original)
            qualify.assert_called_once()

    def test_disabled_native_profile_never_installs_projection(self):
        from vllm_mach.mxfp6.gdn import dense_norm, dense_packed
        with patch.object(legacy, '_ENABLED', False), \
                patch.object(dense_norm, 'prepare'), \
                patch.object(dense_packed, 'prepare'), \
                patch.object(direct_norm, 'prepare') as install:
            legacy.prepare(types.SimpleNamespace())
        install.assert_not_called()

    def test_every_prepared_layer_binds_before_per_device_warmup_return(self):
        # Execute the real legacy.prepare flow with opaque metadata and no math.
        torch = types.SimpleNamespace(float16=object(), bfloat16=object())

        class Norm:
            def __init__(self):
                self.weight = types.SimpleNamespace(shape=(128,), dtype=torch.bfloat16)
                self.bias = self.group_size = None
                self.eps, self.norm_before_gate, self.activation = 1e-6, True, 'silu'
                self._forward_method = self.forward_native

            def forward_native(self, x, z):
                raise AssertionError('No norm math in this lifecycle test')

        modules = {
            'torch': torch,
            'vllm.model_executor.layers.layernorm': types.SimpleNamespace(RMSNormGated=Norm),
            'vllm.third_party.flash_linear_attention.ops': types.SimpleNamespace(
                layernorm_guard=types.SimpleNamespace()),
        }
        prepared = []

        def install(layer):
            self.assertTrue(layer._mach_native_legacy)
            self.assertIs(layer.norm._forward_method.__code__,
                          direct_norm.nested_code(legacy.prepare, 'forward'))
            self.assertIs(layer._rms_norm_gated_cuda.__code__,
                          direct_norm.nested_code(legacy.prepare, 'decode_norm'))
            prepared.append(layer)

        with contextlib.ExitStack() as stack:
            for name in ('dense_norm', 'dense_packed', 'mixed_fp16', 'packed'):
                stub = types.SimpleNamespace(prepare=lambda layer: None)
                modules[gdn.__name__ + '.' + name] = stub
                stack.enter_context(patch.object(gdn, name, stub, create=True))
            stack.enter_context(patch.dict('sys.modules', modules))
            stack.enter_context(patch.object(legacy, '_ENABLED', True))
            stack.enter_context(patch.object(legacy, '_WARMED', {0}))
            stack.enter_context(patch.object(legacy, '_STATS', {'prepared': []}))
            stack.enter_context(patch.object(direct_norm, 'prepare', side_effect=install))
            for index in range(2):
                layer = types.SimpleNamespace(
                    prefix=f'model.layers.{index}.linear_attn', norm=Norm(),
                    _mach_gdn_prepared=True, tp_size=2, num_k_heads=16, num_v_heads=32,
                    in_proj_ba=types.SimpleNamespace(weight=types.SimpleNamespace(
                        shape=(32, 2048), device=types.SimpleNamespace(index=0))),
                    get_state_dtype=lambda: (torch.bfloat16, torch.float16),
                    gdn_prefill_backend='triton', layer_norm_epsilon=1e-6,
                    vllm_config=types.SimpleNamespace(compilation_config=
                        types.SimpleNamespace(custom_ops=[])),
                    _rms_norm_gated_cuda=lambda *args: None)
                legacy.prepare(layer)
            self.assertEqual(len(prepared), 2)
            legacy.prepare(prepared[0])
            self.assertEqual(len(prepared), 2, 'Native prepare must remain idempotent')
