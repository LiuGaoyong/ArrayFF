"""Thin wrappers around :mod:`autoray` used by every numerical module.

``arrayff`` talks to arrays exclusively through :func:`do`. The wrapper exists
so that (a) the ``advect`` module alias is registered exactly once, before any
dispatch happens, and (b) error messages point at arrayff instead of autoray.
"""

from __future__ import annotations

from typing import Any

import autoray as ar
from autoray.autoray import register_module_alias

from arrayff.utils.backend import infer_backend

_ADVECT_ALIAS_REGISTERED = False


def register_advect_alias() -> None:
    """Teach autoray how to dispatch on :mod:`advect` traced arrays.

    ``autoray`` already infers the backend name ``"advect"`` from the tracer
    class, but it has no module mapping for it. Without this alias every
    ``do(...)`` call inside an ``advect`` trace raises::

        ImportError: autoray couldn't find function 'sum' for backend 'advect'

    The mapping is registered lazily and only once; importing arrayff does not
    import advect itself.
    """
    global _ADVECT_ALIAS_REGISTERED
    if _ADVECT_ALIAS_REGISTERED:
        return
    register_module_alias("advect", "advect.numpy")
    _ADVECT_ALIAS_REGISTERED = True


def do(fn: str, *args: Any, like: str | None = None, **kwargs: Any) -> Any:
    """Dispatch ``fn`` to the backend of the given arrays.

    Parameters
    ----------
    fn:
        Name of the operation, e.g. ``"sum"`` or ``"linalg.solve"``.
    like:
        Force a backend instead of inferring it from ``args``.
    """
    register_advect_alias()
    return ar.do(fn, *args, like=like, **kwargs)


def require_same_backend(*arrays: Any) -> str:
    """Return the common backend of ``arrays`` or raise ``TypeError``."""
    return infer_backend(*arrays)


def to_backend(array: Any, backend: str) -> Any:
    """Convert ``array`` to ``backend`` while preserving autograd state."""
    register_advect_alias()
    if backend == "numpy":
        import numpy as np

        return np.asarray(array)
    if backend == "torch":
        import torch

        if isinstance(array, torch.Tensor):
            return array
        return torch.as_tensor(array)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(array)
    if backend == "advect":
        import advect

        return advect.asarray(array)
    raise ValueError(f"Unknown backend: {backend!r}")
