# SPDX-License-Identifier: Apache-2.0
"""Qualified M4 presence head graph; sampling and history updates remain outside."""
from threading import Lock


def identity(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


_SHARD_FIELDS = ('org_vocab_start_index', 'org_vocab_end_index', 'num_org_vocab_padding',
                 'num_added_elements_padded', 'num_elements_padded', 'num_org_elements',
                 'num_org_elements_padded', 'added_vocab_start_index', 'num_added_elements')

def shard_identity(shard):
    return tuple(getattr(shard, name, None) for name in _SHARD_FIELDS)


class PresenceHeadGraph:
    def __init__(self, proxy, head, hidden, presence, counts, slots, *, top_k, temperature):
        import torch
        from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import _HybridHeadGraphs
        self.torch, self.proxy, self.head = torch, proxy, head
        self.original = proxy._get_local_topk
        self.original_function = self.original.__func__
        self.state = proxy._get_hybrid_lm_head_state(head)
        self.signature_fn = _HybridHeadGraphs.signature
        self.signature = self.signature_fn(self.state)
        self.head_graphs = self.state._head_graphs
        self.weight, self.weight_identity = head.weight, identity(head.weight)
        self.shard = head.shard_indices
        assert self.shard.num_elements_padded == 248320
        self.shard_fields = shard_identity(self.shard)
        self.counts, self.counts_identity = counts, identity(counts)
        self.top_k, self.temperature = top_k, temperature
        self.stream = torch.cuda.current_stream(hidden.device)
        assert self.head_graphs is not None and self.stream == self.head_graphs.stream
        assert not torch.cuda.is_current_stream_capturing()
        assert proxy.hybrid_lm_head_enabled and proxy.hybrid_lm_head_row_mask is None
        assert proxy.soft_cap is None and proxy.scale == 1
        assert tuple(hidden.shape) == (4, 2048) and hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
        assert self.state.candidate_count_for_topk(top_k) == 128
        assert self.state.input_size == 2048 and self.state.output_size == 248320
        assert tuple(presence.shape) == tuple(slots.shape) == (4,)
        assert presence.is_contiguous() and slots.is_contiguous()
        assert counts.ndim == 2 and counts.shape[1] == 248320
        assert all(t.device == hidden.device for t in (presence, counts, slots, head.weight))
        self.hidden, self.presence, self.slots = hidden.clone(), presence.clone(), slots.clone()
        self.inputs = [(tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in (hidden, presence, slots)]
        self.lock = Lock()
        def body():
            return self.original(head, self.hidden, top_k=top_k, temperature=temperature,
                presence_penalties=self.presence, output_token_counts=counts,
                presence_request_indices=self.slots)
        with torch.inference_mode():
            body()  # Normal already-warmed coarse/selector route first.
            self.stream.synchronize()  # Preparation only.
            self.graph = torch.cuda.CUDAGraph()
            # Capture the original bound callable. Existing private replay
            # helpers execute their original operations during graph capture.
            with torch.cuda.graph(self.graph):
                self.outputs = body()
        assert self.outputs[2] == 0 and self.outputs[3] is None

    def get(self, lm_head, hidden_states, *, top_k, temperature,
            presence_penalties=None, output_token_ids=None, output_token_counts=None,
            presence_request_indices=None, output_unique_token_ids=None,
            num_output_unique_tokens=None, embedding_bias=None, _hybrid_enabled=None):
        torch = self.torch
        def fallback():
            return self.original(lm_head, hidden_states, top_k=top_k, temperature=temperature,
                presence_penalties=presence_penalties, output_token_ids=output_token_ids,
                output_token_counts=output_token_counts, presence_request_indices=presence_request_indices,
                output_unique_token_ids=output_unique_token_ids, num_output_unique_tokens=num_output_unique_tokens,
                embedding_bias=embedding_bias, _hybrid_enabled=_hybrid_enabled)
        if presence_penalties is None or tuple(hidden_states.shape) != (4, 2048):
            return fallback()
        if (lm_head is not self.head or lm_head.weight is not self.weight
                or identity(lm_head.weight) != self.weight_identity
                or lm_head.shard_indices is not self.shard or shard_identity(self.shard) != self.shard_fields
                or self.proxy._get_hybrid_lm_head_state(lm_head) is not self.state
                or self.signature_fn(self.state) != self.signature
                or self.state._head_graphs is not self.head_graphs
                or (self.state.input_size, self.state.output_size, self.state.candidates) != (2048, 248320, 128)
                or not self.proxy.hybrid_lm_head_enabled or self.proxy.hybrid_lm_head_row_mask is not None
                or self.proxy.soft_cap is not None or self.proxy.scale != 1
                or top_k != self.top_k or temperature != self.temperature
                or presence_penalties is None or presence_request_indices is None
                or output_token_counts is not self.counts or identity(output_token_counts) != self.counts_identity
                or output_token_ids is not None or output_unique_token_ids is not None
                or num_output_unique_tokens is not None or embedding_bias is not None or _hybrid_enabled is not None
                or not torch.is_inference_mode_enabled() or torch.cuda.is_current_stream_capturing()
                or torch.cuda.current_stream(hidden_states.device) != self.stream):
            return fallback()
        actual = (hidden_states, presence_penalties, presence_request_indices)
        if ([(tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in actual] != self.inputs
                or not self.lock.acquire(blocking=False)):
            return fallback()
        try:
            self.hidden.copy_(hidden_states)
            self.presence.copy_(presence_penalties)
            self.slots.copy_(presence_request_indices)
            self.graph.replay()
            # Independent output ownership across two in-flight callers.
            values, indices = self.outputs[:2]
            result = values.clone(), indices.clone(), self.outputs[2], self.outputs[3]
            return result
        finally:
            self.lock.release()


_TEMPERATURE = 0.699999988079071  # Original float32 sampling metadata value.
_STATE = '_mach_2b_presence_graph'


def install(worker, state):
    """Prepare inside the existing 2B head warmup, before serving requests."""
    contract = getattr(worker, '_mach_2b_contract', {})
    if (contract.get('mode') != 'production'
            or contract.get('ssm_dtype') != 'float32'
            or not contract.get('hybrid_greedy_head')
            or state._head_graphs is None):
        return False
    if hasattr(worker, _STATE):
        return False
    from importlib.metadata import version
    if version('vllm').split('+', 1)[0] != '0.33.0':
        raise RuntimeError('2B presence graph requires the selected vLLM runtime')
    import torch
    from vllm_mach.mxfp6.sampling import SamplingHead
    from vllm_mach.mxfp6.hybrid_nvfp4_lm_head import HybridNvfp4LmHead
    runner = worker.model_runner
    model = getattr(runner.model, 'language_model', runner.model)
    head, processor = model.lm_head, model.logits_processor
    if (type(state) is not HybridNvfp4LmHead or state.backend != 'b12x'
            or (state.input_size, state.output_size, state.candidates, state.max_rows) != (2048, 248320, 128, 64)
            or getattr(runner, '_mach_sampling_proxy', None) is not None
            or torch.cuda.is_current_stream_capturing()):
        return False
    # Same proxy ownership and bookkeeping as try_sample's original cold setup;
    # no sampling, seed generation, histogram or request-state update occurs.
    proxy = SamplingHead(processor, state)
    counts = runner.sampler.penalties_state.output_bin_counts
    if (counts.ndim != 2 or counts.shape[0] < 4 or counts.shape[1] != 248320
            or counts.dtype != torch.int32 or not counts.is_contiguous()
            or counts.device != head.weight.device):
        return False
    with torch.inference_mode():
        hidden = torch.zeros((4, 2048), dtype=torch.bfloat16, device=head.weight.device)
        presence = torch.zeros(4, dtype=torch.float32, device=head.weight.device)
        slots = torch.arange(4, dtype=torch.int64, device=head.weight.device)
        graph = PresenceHeadGraph(proxy, head, hidden, presence, counts, slots,
                                  top_k=20, temperature=_TEMPERATURE)
    proxy._get_local_topk = graph.get
    runner._mach_sampling_proxy = proxy
    runner._mach_sampling_head = (head, head.weight.data_ptr())
    embedding = getattr(getattr(model, 'model', None), 'embed_tokens', None)
    runner._mach_sampling_tied_embedding = (
        embedding
        if embedding is not None
        and embedding.weight.data_ptr() == head.weight.data_ptr()
        else None
    )
    runner._mach_sampling_stats = {'topk_calls': 0, 'full_calls': 0, 'rows': 0}
    setattr(worker, _STATE, graph)
    return True


def receipt(worker):
    graph = getattr(worker, _STATE, None)
    if graph is None:
        return {'installed': False}
    import hashlib
    from pathlib import Path
    proxy = worker.model_runner._mach_sampling_proxy
    return {'installed': True, 'rows': 4, 'top_k': 20, 'temperature': _TEMPERATURE,
            'actual_proxy_entry': getattr(proxy._get_local_topk, '__self__', None) is graph,
            'owned_outputs': True, 'live_counts': True,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
