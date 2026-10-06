# SPDX-License-Identifier: Apache-2.0

"""Optional native MXFP8 integration, with no CUDA work at import time.

The worker module exposes the installation boundary used by both the native
opt-in plugin and a complete model profile. Native library loading remains
inside ``install_worker_backend``, after the worker initializes its device.
"""

from .worker import install_worker_backend, install_worker_hook, verify_worker_execution

__all__ = ["install_worker_backend", "install_worker_hook", "verify_worker_execution"]
