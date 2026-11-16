# SPDX-License-Identifier: Apache-2.0
"""Bounded original-kernel prefill graphs behind the unchanged public API.

No kernel, floating-point operation, KV ownership, or compiler behavior changes.
The bound worker owns immutable execution configuration between calls;
it does not emulate arbitrary concurrent Python mutation inside a CUDA graph.
"""
import importlib
import threading

INPUTS = ('q', 'k', 'v', 'g', 'beta', 'initial_state',
          'cu_seqlens', 'chunk_indices', 'chunk_offsets')
TOKEN_INPUTS = INPUTS[:5]
T_CAPACITY = 2048
ALLOWED_KEYS = frozenset(((1, 29), (1, 31), (1, 32), (2, 33)))
HELPERS = ('chunk_local_cumsum', 'chunk_scaled_dot_kkt_fwd', 'solve_tril',
           'recompute_w_u_fwd', 'chunk_gated_delta_rule_fwd_h', 'chunk_fwd_o')


class GraphDispatch:
    def __init__(self, chunk, installed, stream):
        import torch
        from vllm_mach.mxfp6.gdn import legacy
        from triton.runtime import jit
        self.torch, self.legacy, self.jit = torch, legacy, jit
        self.chunk, self.installed, self.stream = chunk, installed, stream
        self.original = chunk.chunk_gated_delta_rule_fwd
        self.device = stream.device
        self.thread = threading.get_ident()
        self.banks = {}
        self.sealed = False
        self.running = False
        # A bound method is stored once so identity checks are meaningful.
        self.run = self._run

    def _key(self, args):
        return (args['initial_state'].shape[0], args['chunk_indices'].shape[0])

    def _spec(self, tensor):
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
                tensor.device, tensor.data_ptr() % 16)

    def _capacity_spec(self, name, tensor, rows):
        """Normalize only T and its canonical leading singleton stride."""
        shape, stride, dtype, device, alignment = self._spec(tensor)
        if name in TOKEN_INPUTS:
            ndim = 3 if name in ('g', 'beta') else 4
            if (len(shape) != ndim or shape[1] != rows
                    or stride[0] != rows * stride[1]):
                return None
            shape = (shape[0], T_CAPACITY, *shape[2:])
            stride = (T_CAPACITY * stride[1], *stride[1:])
        elif name == 'core_attn_out':
            if len(shape) != 3 or shape[0] != rows:
                return None
            shape = (T_CAPACITY, *shape[1:])
        return shape, stride, dtype, device, alignment

    def _basic(self, args):
        t = self.torch
        if (not t.is_inference_mode_enabled() or t.is_grad_enabled()
                or type(args['scale']) is not float or args['scale'] != 128**-0.5
                or args['output_final_state'] is not True):
            return False
        for name in INPUTS + ('core_attn_out',):
            value = args[name]
            if (type(value) is not t.Tensor or value.device != self.device
                    or not value.is_contiguous() or value.data_ptr() % 16):
                return False
        q, k, v = (args[n] for n in ('q', 'k', 'v'))
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != (8, 128):
            return False
        T = q.shape[1]
        initial, ci, cu, co = (args[n] for n in
                              ('initial_state', 'chunk_indices', 'cu_seqlens', 'chunk_offsets'))
        if initial.ndim != 4 or initial.shape[1:] != (16, 128, 128):
            return False
        N = initial.shape[0]
        return (k.shape == q.shape and v.shape == (1, T, 16, 128)
                and q.dtype == k.dtype == v.dtype == t.bfloat16
                and args['g'].shape == args['beta'].shape == (1, T, 16)
                and args['g'].dtype == args['beta'].dtype == t.float32
                and initial.dtype == t.float16
                and cu.shape == co.shape == (N + 1,)
                and cu.dtype == ci.dtype == t.int32 and co.dtype == t.int64
                and ci.ndim == 2 and ci.shape[1] == 2
                and args['core_attn_out'].shape == (T, 16, 128)
                and args['core_attn_out'].dtype == t.bfloat16)

    def prepare(self, args):
        """Called only before timing, with explicit GPU chunk metadata."""
        t = self.torch
        assert not self.sealed and self.chunk.chunk_gated_delta_rule_fwd is self.original
        assert threading.get_ident() == self.thread and self._basic(args)
        assert t.cuda.current_stream(self.device).cuda_stream == self.stream.cuda_stream
        assert not t.cuda.is_current_stream_capturing()
        rows = args['q'].shape[1]
        assert 0 < rows <= T_CAPACITY
        key = self._key(args)
        assert key in ALLOWED_KEYS
        if key in self.banks:
            return dict(created=False, key=list(key))
        assert len(self.banks) < 4, 'The component has exactly four bounded geometries'
        capture_stream = self.capture_stream
        capture_stream.wait_stream(self.stream)
        with t.cuda.stream(capture_stream):
            stage = {}
            for name in INPUTS:
                value = args[name]
                if name in TOKEN_INPUTS:
                    stage[name] = t.empty((1, T_CAPACITY, *value.shape[2:]),
                                          dtype=value.dtype, device=value.device)
                    stage[name][:, :rows].copy_(value, non_blocking=True)
                else:
                    stage[name] = value.clone()
            output = args['core_attn_out']
            stage['core_attn_out'] = t.empty((T_CAPACITY, *output.shape[1:]),
                                            dtype=output.dtype, device=output.device)
            stage['scale'] = args['scale']
            stage['output_final_state'] = True
            # Original helper receives all three precomputed metadata tensors.
            # It cannot enter its missing-metadata CPU/H2D preparation branch.
            warmed = self.original(**stage)
            capture_stream.synchronize()
            del warmed
            graph = t.cuda.CUDAGraph()
            # No shared graph pool: arbitrary key order is safe, and all layers
            # reuse this key's private bank serially on the bound stream.
            with t.cuda.graph(graph, stream=capture_stream):
                outputs = self.original(**stage)
        self.stream.wait_stream(capture_stream)
        assert len(outputs) == 7 and outputs[1].data_ptr() == stage['core_attn_out'].data_ptr()
        self.banks[key] = dict(stage=stage, graph=graph, outputs=outputs,
                               specs={name: self._spec(stage[name])
                                      for name in INPUTS + ('core_attn_out',)})
        return dict(created=True, key=list(key))

    def seal(self):
        """Bind the actual fixed component owner after receipt hooks detach."""
        from vllm_mach.mxfp6.gdn.qualified_cache import GuardedCache
        assert not self.sealed and len(self.banks) == 4
        assert self.chunk.chunk_gated_delta_rule_fwd is self.original
        assert self.legacy.enabled()
        self.helpers = tuple((name, getattr(self.chunk, name)) for name in HELPERS)
        self.chunk_constants = (self.chunk.SUPPRESS_LEVEL, self.chunk.FLA_CHUNK_SIZE)
        self.original_code = self.original.__code__
        self.owners = []
        for row, leaf, original, tuner in self.installed:
            assert leaf.run is original and not leaf.pre_run_hooks
            cache = tuner.cache
            assert isinstance(cache, GuardedCache)
            key = tuple(row['key'])
            choice = cache.qualified.choose(cache.view, cache.row, key)
            assert cache.raw.get(key) is choice
            module = importlib.import_module(row['module'])
            root = getattr(module, row['function'])
            chain = []
            node = root
            while node is not leaf:
                chain.append((node, node.fn))
                node = node.fn
            self.owners.append(dict(module=module, name=row['function'], root=root,
                chain=tuple(chain), leaf=leaf, leaf_run=original, tuner=tuner,
                tuner_run=tuner.run, cache=cache, qualified=cache.qualified,
                choose=type(cache.qualified).choose, key=key, choice=choice,
                src=leaf.src, hash=leaf.hash, debug=leaf.debug, globals=leaf.used_global_vals,
                global_items=tuple(leaf.used_global_vals.items()),
                device_cache=leaf.device_caches[self.device.index],
                compiled=tuple(leaf.device_caches[self.device.index][0].items())))
        assert len(self.owners) == 6
        self.knobs = (self.jit.knobs.runtime.debug,
                      self.jit.knobs.compilation.instrumentation_mode)
        self.launch_hooks = tuple(getattr(self.jit.knobs.runtime, name)
                                  for name in ('launch_enter_hook', 'launch_exit_hook'))
        self.sealed = True

    def _empty_launch_hooks(self):
        for name, owner in zip(('launch_enter_hook', 'launch_exit_hook'), self.launch_hooks):
            current = getattr(self.jit.knobs.runtime, name)
            if current is not owner:
                return False
            if current is not None and not (
                    type(current) is self.jit.knobs.HookChain and not current.calls):
                return False
        return True

    def _owners_ok(self):
        if (not self.legacy.enabled() or self.original.__code__ is not self.original_code
                or (self.chunk.SUPPRESS_LEVEL, self.chunk.FLA_CHUNK_SIZE) != self.chunk_constants
                or any(getattr(self.chunk, name) is not owner for name, owner in self.helpers)
                or self.knobs != (self.jit.knobs.runtime.debug,
                                  self.jit.knobs.compilation.instrumentation_mode)
                or self.jit.knobs.runtime.add_stages_inspection_hook is not None
                or not self._empty_launch_hooks()):
            return False
        missing = object()
        for s in self.owners:
            leaf, tuner, cache = s['leaf'], s['tuner'], s['cache']
            if (getattr(s['module'], s['name']) is not s['root']
                    or any(node.fn is not child for node, child in s['chain'])
                    or leaf.run is not s['leaf_run'] or leaf.pre_run_hooks
                    or tuner.run is not s['tuner_run'] or tuner.cache is not cache
                    or cache.qualified is not s['qualified']
                    or type(cache.qualified).choose is not s['choose']
                    or leaf.src is not s['src'] or leaf.hash != s['hash']
                    or leaf.debug != s['debug']
                    or leaf.device_caches.get(self.device.index) is not s['device_cache']
                    or any(s['device_cache'][0].get(key) is not compiled
                           for key, compiled in s['compiled'])
                    or leaf.used_global_vals is not s['globals']
                    or len(leaf.used_global_vals) != len(s['global_items'])):
                return False
            try:
                choice = cache.qualified.choose(cache.view, cache.row, s['key'])
            except RuntimeError:
                return False  # Original fallback retains the actual original error.
            if choice is not s['choice'] or cache.raw.get(s['key']) is not choice:
                return False
            for key, expected in s['global_items']:
                if leaf.used_global_vals.get(key) is not expected:
                    return False
                value, scope = expected
                if scope.get(key[0], missing) != value:
                    return False
        return True

    def _fallback(self, args, reason):
        return self.original(**args)

    def _run(self, q, k, v, g, beta, scale, initial_state, output_final_state,
             cu_seqlens=None, chunk_indices=None, chunk_offsets=None, core_attn_out=None):
        t = self.torch
        args = dict(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    initial_state=initial_state, output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets, core_attn_out=core_attn_out)
        if (not self.sealed or self.running or threading.get_ident() != self.thread
                or t.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream
                or t.cuda.is_current_stream_capturing()
                or not t.is_inference_mode_enabled() or t.is_grad_enabled()
                or type(scale) is not float or scale != 128**-0.5
                or output_final_state is not True
                or type(q) is not t.Tensor or q.ndim != 4
                or type(initial_state) is not t.Tensor or initial_state.ndim != 4
                or type(chunk_indices) is not t.Tensor or chunk_indices.ndim != 2):
            return self._fallback(args, 'scope_or_layout')
        rows = q.shape[1]
        if not 0 < rows <= T_CAPACITY:
            return self._fallback(args, 'T_capacity')
        key = self._key(args)
        bank = self.banks.get(key)
        if bank is None:
            return self._fallback(args, 'unknown_geometry')
        # Retain the startup dtype/device/alignment/shape/stride contract.
        # Only T and its canonical leading singleton stride are normalized.
        if any(type(args[name]) is not t.Tensor
               or self._capacity_spec(name, args[name], rows) != bank['specs'][name]
               for name in INPUTS + ('core_attn_out',)):
            return self._fallback(args, 'layout_specialization')
        if not self._owners_ok():
            return self._fallback(args, 'owner_config_or_globals')
        self.running = True
        try:
            # Fresh public final_state preserves cross-call output lifetime.
            final = t.empty_like(bank['outputs'][3])
            for name in INPUTS:
                target = bank['stage'][name]
                if name in TOKEN_INPUTS:
                    target = target[:, :rows]
                target.copy_(args[name], non_blocking=True)
            bank['graph'].replay()
            core_attn_out.copy_(bank['outputs'][1][:, :rows].view_as(core_attn_out),
                                non_blocking=True)
            final.copy_(bank['outputs'][3], non_blocking=True)
            out = core_attn_out.view(1, rows, 16, 128)
            internal = bank['outputs']
            # The unchanged public forward only lets o and final_state escape.
            # g/A (and optional suppressed intermediates) remain bank-private.
            return (internal[0][:, :rows], out, internal[2][:, :rows], final,
                    internal[4][:, :rows] if internal[4] is not None else None,
                    internal[5],
                    internal[6][:, :rows] if internal[6] is not None else None)
        finally:
            self.running = False

    def enable(self):
        assert self.sealed
        assert self.chunk.chunk_gated_delta_rule_fwd in (self.original, self.run)
        self.chunk.chunk_gated_delta_rule_fwd = self.run

    def disable(self):
        assert self.chunk.chunk_gated_delta_rule_fwd in (self.original, self.run)
        self.chunk.chunk_gated_delta_rule_fwd = self.original
