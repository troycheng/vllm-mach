# SPDX-License-Identifier: Apache-2.0
"""Keep the qualified native Dense prefill backend and FLA selections."""
import functools
import hashlib
import importlib
import json
from pathlib import Path
from types import MethodType

from . import dense_prep

_DATA = Path(__file__).parents[1] / 'profile'
_ENABLED = False
_INSTALLED = False
_STATS = {'calls': {}, 'choices': {}}


def resolve_backend(config, requested):
    global _ENABLED
    eligible = (dense_prep.model_eligible(config)
                and config.cache_config.mamba_ssm_cache_dtype in ('float32', 'float16')
                and dense_prep._sm120())
    additional = config.additional_config
    explicit = isinstance(additional, dict) and 'gdn_prefill_backend' in additional
    selected = 'triton' if eligible and not explicit else requested
    _ENABLED = eligible and str(selected).strip().lower() == 'triton'
    if _ENABLED:
        install()
    return selected


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from triton import Config
    from triton.runtime.autotuner import Autotuner

    gates = json.loads((_DATA / 'legacy_sources.json').read_text())
    for filename, digest in gates.items():
        module = importlib.import_module(
            'vllm.third_party.flash_linear_attention.ops.' + filename[:-3])
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Dense FLA source changed: ' + filename)
    rows = json.loads((_DATA / 'dense_fla_choices.json').read_text())['kernels']
    bindings = {}
    for row in rows:
        bindings.setdefault((row['module'], row['function']), []).append(row)
    for (module_name, function), choices in bindings.items():
        tuner = getattr(importlib.import_module(module_name), function)
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        original = tuner.run

        @functools.wraps(original)
        def run(self, *args, _rows=tuple(choices), _original=original, **kwargs):
            values = {**dict(zip(self.arg_names, args)), **kwargs}
            bound = {k: v for k, v in values.items() if k in self.arg_names}
            key = [bound[k] for k in self.keys if k in bound]
            key.extend(str(v.dtype) for v in bound.values() if hasattr(v, 'dtype'))
            row = next((r for r in _rows if key == r['key']), None)
            match = _ENABLED and row is not None
            if match:
                choice = Config(**row['selected'])
                if choice not in self.configs or (
                        tuple(key) in self.cache and self.cache[tuple(key)] != choice):
                    raise RuntimeError('Dense qualified FLA selection changed')
                self.cache[tuple(key)] = choice
            result = _original(*args, **kwargs)
            if match:
                name = row['function']
                _STATS['calls'][name] = _STATS['calls'].get(name, 0) + 1
                selected = dict(key=key, selected=row['selected'])
                if hasattr(result, 'hash'):
                    selected['hash'] = result.hash
                _STATS['choices'][name] = selected
            return result
        tuner.run = MethodType(run, tuner)
    _INSTALLED = True


def receipt():
    return dict(enabled=_ENABLED, installed=_INSTALLED, **_STATS,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                choices_sha256=hashlib.sha256(
                    (_DATA / 'dense_fla_choices.json').read_bytes()).hexdigest(),
                scope='Native Dense H8/HV24 FP32/FP16 prefill only; explicit backend preserved',
                counters='Completed host FLA calls, not warmup admission or graph replays')
