# SPDX-License-Identifier: Apache-2.0
"""Qualified FLA cache policy; original Triton run builds the sole key."""
from collections.abc import MutableMapping
from functools import wraps


class RawTunerView:
    """Use the existing backing dict; choose() cannot recurse into the facade."""
    def __init__(self, tuner, raw):
        self.tuner, self.cache = tuner, raw

    def __getattr__(self, name):
        return getattr(self.tuner, name)


class GuardedCache(MutableMapping):
    def __init__(self, tuner, row, qualified, enabled):
        if type(tuner.cache) is not dict:
            raise TypeError('Candidate requires the original upstream dict cache')
        self.raw = tuner.cache
        self.row, self.qualified, self.enabled = row, qualified, enabled
        self.view = RawTunerView(tuner, self.raw)
        self.first_real_lookup = None
        self.raw_contains_depth = 0
        self.qualification_error = None

    def __contains__(self, key):
        # This is the existing Autotuner.run `if key not in self.cache` lookup.
        # No seed, argument binding, dtype traversal or replacement JIT launch.
        if (not self.raw_contains_depth and self.enabled()
                and key == tuple(self.row['key'])):
            try:
                choice = self.qualified.choose(self.view, self.row, key)
            except BaseException as error:
                self.qualification_error = error
                raise
            self.raw[key] = choice
            if self.first_real_lookup is None:
                self.first_real_lookup = key
        return key in self.raw

    def __getitem__(self, key):
        return self.raw[key]

    def __setitem__(self, key, value):
        self.raw[key] = value

    def __delitem__(self, key):
        del self.raw[key]

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def clear(self):
        self.raw.clear()


def attach(tuner, row, qualified, enabled, original, baseline):
    cache = GuardedCache(tuner, row, qualified, enabled)
    tuner.cache = cache

    @wraps(original)
    def run(*args, **kwargs):
        # Upstream skips its key/cache lookup if configs shrinks to <=1.
        # Preserve current policy's mutation/qualified-key check in that rare
        # case, rather than silently losing it. Normal six-tuner calls form
        # no key here; the original run and its hooks/kwargs remain unchanged.
        prior_nargs = tuner.nargs
        cache.qualification_error = None
        try:
            if enabled() and len(tuner.configs) <= 1:
                cache.raw_contains_depth += 1
                try:
                    return baseline(*args, **kwargs)
                finally:
                    cache.raw_contains_depth -= 1
            return original(*args, **kwargs)
        except BaseException as error:
            # Existing policy rejects before original.run assigns nargs. Only
            # restore that qualification-error case; leaf/prehook errors retain
            # exactly the original upstream partial state.
            if cache.qualification_error is error:
                tuner.nargs = prior_nargs
            raise
        finally:
            cache.qualification_error = None

    return run, cache
