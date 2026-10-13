"""BF16 ordered mixed decode must preserve stock's per-token state rounding."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize('m', [1, 4, 32, 64, 160])
@torch.inference_mode()
def test_mixed_decode(m):
    from vllm_mach.mxfp8.two_b.gdn import ordered_allm_triton as op
    from vllm_mach.mxfp8.two_b.gdn import mixed_ordered_triton as mixed
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update as stock,
    )
    torch.manual_seed(9)
    base = torch.randn(m + 2, 16, 128, 128, device='cuda', dtype=torch.bfloat16) * .1
    ref = base.clone()
    ids = torch.arange(1, m + 1, device='cuda', dtype=torch.int32)
    cu = torch.arange(m + 1, device='cuda', dtype=torch.int32)
    scratch = op.DeferredScratch(**{
        k: torch.zeros(s, device='cuda', dtype=torch.int32 if k == 'age' else torch.float32)
        for k, s in op.required_shapes(m + 2, 4).items()})
    out = torch.empty(m, 1, 16, 128, device='cuda', dtype=torch.bfloat16)
    al = torch.randn(16, device='cuda') - 2
    dt = torch.randn(16, device='cuda', dtype=torch.bfloat16)
    for step in range(64):
        qkv = (torch.randn(m, 8192, device='cuda', dtype=torch.bfloat16) * .1)[:, :6144]
        a = torch.randn(m, 16, device='cuda', dtype=torch.bfloat16)
        b = torch.randn_like(a)
        mixed._ordered_decode[(4, 16, m)](
            qkv, a, b, al, dt, out, base, ids,
            scratch.pending_k, scratch.pending_d, scratch.coeff,
            scratch.prefix, scratch.age,
            qkv.stride(0), a.stride(0), b.stride(0), base.stride(0),
            1, m + 2, 4, 128**-.5, 16, 16, 128, 128, 32, 4,
            num_warps=4, num_stages=3)
        q, k, v = [x.view(1, m, 16, 128) for x in qkv.split(2048, -1)]
        expected, _ = stock(al, a, b, dt, q, k, v, initial_state=ref,
                            cu_seqlens=cu, ssm_state_indices=ids,
                            use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(out.transpose(0, 1), expected, atol=0, rtol=0)
        if step % 7 == 3:
            op.materialize_slots(base, ids, scratch)
            torch.testing.assert_close(base, ref, atol=0, rtol=0)
    op.materialize_slots(base, ids, scratch)
    torch.testing.assert_close(base, ref, atol=0, rtol=0)


@pytest.mark.parametrize('heads', [16, 32])
@torch.inference_mode()
def test_small_large_graph_handoffs(heads):
    """Share W4 history across large Triton, small Gluon, full stores and stock."""
    import importlib
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode as stock,
    )
    name = ('vllm_mach.mxfp8.two_b.gdn.dispatch' if heads == 16 else
            'vllm_mach.mxfp8.gdn.ordered_full_m4_m16_deferred_dispatch')
    op = importlib.import_module(name)
    torch.manual_seed(731)
    pages, maximum, width = 132, 128, 4096 + heads * 128
    backing = torch.randn(pages, heads * 16384 + 128, device='cuda',
                          dtype=torch.bfloat16) * .1
    original = backing.clone()
    base = backing[:, :heads * 16384].view(pages, heads, 128, 128)
    ref = base.clone()
    qkv = torch.randn(maximum, width + heads * 128, device='cuda',
                      dtype=torch.bfloat16)[:, :width]
    a = torch.randn(maximum, heads, device='cuda', dtype=torch.bfloat16)
    b = torch.randn_like(a)
    al = torch.randn(heads, device='cuda') - 2
    dt = torch.randn(heads, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(maximum, 1, heads, 128, device='cuda', dtype=torch.bfloat16)
    expected = torch.empty_like(out)
    ids = torch.arange(1, maximum + 1, device='cuda', dtype=torch.int32)
    scratch = op.DeferredScratch(**{
        k: torch.zeros(s, device='cuda', dtype=torch.int32 if k == 'age' else torch.float32)
        for k, s in op.required_shapes(pages, 4).items()})
    def call(m):
        op.decode(qkv[:m], a[:m], b[:m], al, dt, base, out[:m], ids[:m], scratch)
    graphs = {}
    for m in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
        call(m)
        graphs[m] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[m]):
            call(m)
    backing.copy_(original)
    scratch.age.zero_()
    ref.copy_(base)
    # Vary age on individual tiles, including invalid slots. Each small path
    # must consume history left by the large path before committing its token.
    schedule = (128, 128, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 3,
                5 if heads == 16 else 3)
    for step in range(96):
        m = schedule[step % len(schedule)]
        qkv.normal_(std=.1)
        a.normal_()
        b.normal_()
        ids.copy_(torch.arange(1, maximum + 1, device='cuda', dtype=torch.int32).roll(step))
        if step % 7 == 2:
            ids[0] = 0
        elif step % 7 == 3:
            ids[0] = -1
        if m == 5:
            from vllm_mach.mxfp8.two_b.gdn import mixed_ordered_triton as mixed
            from vllm.third_party.flash_linear_attention.ops import (
                fused_sigmoid_gating_delta_rule_update as stock_mixed,
            )
            # Stock mixed leaves invalid-row output undefined; compare valid rows.
            ids.copy_(torch.arange(1, maximum + 1, device='cuda', dtype=torch.int32).roll(step))
            mixed._ordered_decode[(4, heads, m)](
                qkv, a, b, al, dt, out, base, ids,
                scratch.pending_k, scratch.pending_d, scratch.coeff,
                scratch.prefix, scratch.age, qkv.stride(0), a.stride(0),
                b.stride(0), base.stride(0), 1, pages, 4, 128**-.5,
                16, heads, 128, 128, 32, 4, num_warps=4, num_stages=3)
            q, k, v = [x.view(1, m, 16, 128) for x in qkv[:m].split(2048, -1)]
            mixed_expected, _ = stock_mixed(
                al, a[:m], b[:m], dt, q, k, v, initial_state=ref,
                cu_seqlens=torch.arange(m + 1, device='cuda', dtype=torch.int32),
                ssm_state_indices=ids[:m], use_qk_l2norm_in_kernel=True)
            expected[:m].copy_(mixed_expected.transpose(0, 1))
        elif m == 3:  # Unsupported pure-decode rows: flush before stock fallback.
            op.materialize_slots(base, ids[:m], scratch)
            stock(qkv[:m], a[:m], b[:m], al, dt, 128**-.5,
                  base, out[:m], ids[:m], True)
        else:
            graphs[m].replay()
        if m != 5:
            stock(qkv[:m], a[:m], b[:m], al, dt, 128**-.5,
                  ref, expected[:m], ids[:m], True)
        torch.testing.assert_close(out[:m], expected[:m], atol=0, rtol=0)
        if step % 17 == 0:
            all_ids = torch.arange(1, pages, device='cuda', dtype=torch.int32)
            op.materialize_slots(base, all_ids, scratch)
            torch.testing.assert_close(base, ref, atol=0, rtol=0)
    op.materialize_slots(base, torch.arange(1, pages, device='cuda', dtype=torch.int32), scratch)
    torch.testing.assert_close(base, ref, atol=0, rtol=0)
    assert torch.equal(backing[:, -128:], original[:, -128:])
    assert torch.equal(base[0], original[0, :heads * 16384].view(heads, 128, 128))
