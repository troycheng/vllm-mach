from types import SimpleNamespace as N
import pytest
from vllm_mach.mxfp6.gdn.mixed_fp16 import _eligible as eligible, _clone_frontend as private_frontend


def tensor(shape, dtype='torch.bfloat16', strides=None):
    return N(shape=shape, dtype=dtype, is_cuda=True, stride=lambda: strides or (1,))


def fixture():
    n = 29
    state = tensor((1478, 16, 128, 128), 'torch.float16', (278528, 16384, 128, 1))
    layer = N(_mach_native_legacy=True, tp_size=2, num_k_heads=16, num_v_heads=32,
              head_k_dim=128, head_v_dim=128, kv_cache=(None, state))
    md = N(num_prefills=1, num_decodes=n, num_decode_tokens=n, num_spec_decodes=0, spec_sequence_masks=None)
    call = dict(q=tensor((1, n, 8, 128)), k=tensor((1, n, 8, 128)), v=tensor((1, n, 16, 128)),
                a=tensor((n, 16)), b=tensor((n, 16)), A_log=tensor((16,), 'torch.float32'),
                dt_bias=tensor((16,)), initial_state=state, cu_seqlens=tensor((n+1,), 'torch.int32'),
                ssm_state_indices=tensor((n+1,), 'torch.int32'), use_qk_l2norm_in_kernel=True)
    return layer, md, call


def test_actual_varlen_metadata_prefix_and_pool_capacity_are_not_quality_specific():
    layer, md, call = fixture()
    assert eligible(layer, md, call, enabled=True, sm120=True)
    assert call['initial_state'].shape[0] == 1478  # synthetic larger capacity; no626slot requirement or actual capacity claim
    assert call['ssm_state_indices'].shape[0] > md.num_decodes  # real mixed metadata includes prefill tail


@pytest.mark.parametrize('change', ['disabled', 'wrong_sm', 'unprepared', 'tp1', 'dense', 'no_prefill',
                                  'spec', 'spec_mask', 'multi_decode', 'too_many_rows', 'nonvarlen',
                                  'FP32state', 'wrong_pool_stride', 'wrong_pool_owner', 'wrong_CU',
                                  'wrong_index_dtype', 'output', 'MTP', 'KDA', 'no_qk_norm'])
def test_unqualified_paths_delegate_stock(change):
    layer, md, call = fixture()
    enabled = sm120 = True
    if change == 'disabled': enabled = False
    if change == 'wrong_sm': sm120 = False
    if change == 'unprepared': layer._mach_native_legacy = False
    if change == 'tp1': layer.tp_size = 1
    if change == 'dense': layer.num_v_heads = 48
    if change == 'no_prefill': md.num_prefills = 0
    if change == 'spec': md.num_spec_decodes = 1
    if change == 'spec_mask': md.spec_sequence_masks = object()
    if change == 'multi_decode': md.num_decode_tokens += 1
    if change == 'too_many_rows': md.num_decodes = 33
    if change == 'nonvarlen': call['q'].shape = (29, 1, 8, 128)
    if change == 'FP32state': call['initial_state'].dtype = 'torch.float32'
    if change == 'wrong_pool_stride': call['initial_state'].stride = lambda: (262144, 16384, 128, 1)
    if change == 'wrong_pool_owner': call['initial_state'] = tensor((1478,16,128,128), 'torch.float16', (278528,16384,128,1))
    if change == 'wrong_CU': call['cu_seqlens'].shape = (29,)
    if change == 'wrong_index_dtype': call['ssm_state_indices'].dtype = 'torch.int64'
    if change == 'output': call['output'] = object()
    if change == 'MTP': call['num_accepted_tokens'] = object()
    if change == 'KDA': call['is_kda'] = True
    if change == 'no_qk_norm': call['use_qk_l2norm_in_kernel'] = False
    assert not eligible(layer, md, call, enabled=enabled, sm120=sm120)


class Heuristic:
    def __init__(self, fn): self.fn = fn
    def __call__(self, value): return self.fn(value)


fused_sigmoid_gating_delta_rule_update_kernel = Heuristic(lambda x: ('stock', x))


def frontend(x, *, marker=None):
    return fused_sigmoid_gating_delta_rule_update_kernel(x), marker


def test_private_clone_preserves_original_globals_defaults_and_aliases():
    original_kernel = fused_sigmoid_gating_delta_rule_update_kernel.fn
    private = private_frontend(frontend, lambda x: ('candidate', x))
    value, marker = object(), object()
    assert private(value, marker=marker) == (('candidate', value), marker)
    assert frontend(value, marker=marker) == (('stock', value), marker)
    assert private.__code__ is frontend.__code__ and private.__kwdefaults__ == frontend.__kwdefaults__
    assert fused_sigmoid_gating_delta_rule_update_kernel.fn is original_kernel and private.__globals__['fused_sigmoid_gating_delta_rule_update_kernel'] is not fused_sigmoid_gating_delta_rule_update_kernel
