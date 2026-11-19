"""Exercise the 0.34 GDN gate and launcher handoff with a fake compiler."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from vllm_mach.mxfp8.two_b import gdn_norm_tile as tile

NAME = 'triton_per_fused__to_copy_add_mean_mul_mxfp8_quantize_pow_rsqrt_silu_view_0'
SOURCE = (Path(__file__).parent / 'fixtures/mxfp8_2b_v034_gated_norm.py').read_text()


class Autotuner:
    def __init__(self, fn, triton_meta, configs, **kwargs):
        self.fn = fn
        self.device_props = triton_meta['device']
        self.triton_meta = {**triton_meta, 'device': self.device_props.index}
        self.configs = configs
        self.grid_mode = 'python'
        self.calls = []
        self.__dict__.update(kwargs)

    def _ensure_kernel_loaded(self):
        self.fn = NS(src=self.fn.src, fn=object())

    def precompile(self):
        self.launchers = [NS(config=config) for config in self.configs]

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.configs[0].kwargs['XBLOCK']


def config(kwargs, **fields):
    return NS(kwargs=kwargs, **fields)


def kernel(*, cached=False):
    return Autotuner(
        NS(src=SOURCE, fn=None if cached else object()),
        triton_meta={
            'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16',
                          'out_ptr1': '*bf16', 'ks0': 'i64', 'xnumel': 'i32',
                          'r0_numel': 'i32', 'XBLOCK': 'constexpr'},
            'device': NS(type='cuda', cc=120, index=0),
            'constants': {},
            'configs': [{(i,): [['tt.divisibility', 16]] for i in (0, 1, 2, 3, 5, 6)}],
            'enable_fp_fusion': True,
            'launch_pdl': False,
            'disable_ftz': False,
            'native_matmul': False,
        },
        configs=[config({'XBLOCK': 32})],
        size_hints={'x': 32768, 'r0_': 128},
        inductor_meta={'grid_type': 'Grid1D', 'dynamic_scale_rblock': True},
        mutated_arg_names=[], reset_to_zero_arg_names=[], optimize_mem=True,
        heuristic_type='persistent_reduction', filename='generated.py',
    )


def test_exact_gated_norm_boundary():
    original = kernel()
    assert tile._matches(original)
    original.fn.src = SOURCE.replace('1e-06', '2e-06')
    assert not tile._matches(original)
    original.fn.src = SOURCE
    original.triton_meta['enable_fp_fusion'] = False
    assert not tile._matches(original)
    original.triton_meta['enable_fp_fusion'] = True
    original.device_props.cc = 121
    assert not tile._matches(original)
    original.device_props.cc = 120
    original.size_hints['x'] = 131072
    assert not tile._matches(original)


@pytest.mark.parametrize('cached', [False, True])
def test_wait_wraps_only_qualified_kernel_and_preserves_other_launches(monkeypatch, cached):
    class AsyncCompile:
        def wait(self, scope):
            scope[NAME] = original
            return 'waited'

    capturing = False
    original = kernel(cached=cached)
    unrelated = kernel()
    changed = kernel()
    changed.triton_meta['launch_pdl'] = True
    metadata = deepcopy(original.triton_meta)
    monkeypatch.setattr(tile, '_INSTALLED', False)
    monkeypatch.setattr(tile, '_COUNTS', Counter())
    monkeypatch.setitem(sys.modules, 'torch', NS(cuda=NS(is_current_stream_capturing=lambda: capturing)))
    monkeypatch.setitem(sys.modules, 'torch._inductor.async_compile', NS(AsyncCompile=AsyncCompile))
    monkeypatch.setitem(sys.modules, 'torch._inductor.runtime.triton_heuristics', NS(CachingAutotuner=Autotuner))
    monkeypatch.setitem(sys.modules, 'triton', NS(Config=config))

    monkeypatch.setattr(tile, 'BASE_PACKAGES', {'vllm': '0.33.0'})
    wait = AsyncCompile.wait
    assert not tile.install()
    assert AsyncCompile.wait is wait
    monkeypatch.setattr(tile, 'BASE_PACKAGES', {'vllm': '0.34.0'})
    assert tile.install()
    assert not tile.install()

    other_name = tile._KERNEL_PREFIX + '1'
    scope = {'unrelated': unrelated, other_name: changed}
    assert AsyncCompile().wait(scope) == 'waited'
    assert scope['unrelated'] is unrelated
    assert scope[other_name] is changed
    selected = scope[NAME]
    variant = selected._variant
    assert variant.fn is original.fn
    assert variant.triton_meta == original.triton_meta == metadata
    assert original.inductor_meta['dynamic_scale_rblock'] is True
    assert variant.inductor_meta['dynamic_scale_rblock'] is False
    assert original.configs[0].kwargs == {'XBLOCK': 32}

    operands = tuple(object() for _ in range(4))
    stream = object()
    for scalars in [(8, 128, 128), (17, 272, 128), (32, 512, 128), (16, 256, 64), (16, 128, 128)]:
        assert selected.run(*operands, *scalars, stream=stream) == 32
        assert original.calls[-1] == (operands + scalars, {'stream': stream})
    assert selected.run(*operands, 16, 256, 128, stream=stream, benchmark_run=True) == 32
    assert selected.run(*operands, 16, 256, 128, stream=stream) == 8
    capturing = True
    assert selected.run(*operands, 16, 256, 128, stream=stream) == 8
    assert variant.calls == [(operands + (16, 256, 128), {'stream': stream})] * 2
    assert tile.receipt()['compiled_launchers'] == tile.receipt()['capture_calls'] == tile.receipt()['eager_calls'] == 1
