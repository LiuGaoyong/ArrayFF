"""Thin wrappers around :mod:`autoray` used by every numerical module.

``arrayff`` talks to arrays exclusively through :func:`do`. The wrapper exists
so that (a) the ``autograd`` module alias is registered exactly once, before
any dispatch happens, and (b) error messages point at arrayff instead of
autoray.
"""

from __future__ import annotations

from typing import Any

import autoray as ar
from autoray.autoray import register_module_alias

from arrayff.utils.backend import infer_backend

_AUTOGRAD_ALIAS_REGISTERED = False


def register_autograd_alias() -> None:
    """Teach autoray how to dispatch on :mod:`autograd` traced arrays.

    ``autoray`` infers the backend name ``"autograd"`` for HIPS autograd
    ``ArrayBox`` objects.  The module mapping is registered lazily and only
    once; importing arrayff does not import autograd itself.
    """
    global _AUTOGRAD_ALIAS_REGISTERED
    if _AUTOGRAD_ALIAS_REGISTERED:
        return
    register_module_alias("autograd", "autograd.numpy")
    _AUTOGRAD_ALIAS_REGISTERED = True


def _contains_autograd(value: Any) -> bool:
    """Return whether ``value`` (recursively) contains autograd boxes."""
    if type(value).__module__.split(".")[0] == "autograd":
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_autograd(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_autograd(item) for item in value.values())
    return False


def do(fn: str, *args: Any, like: str | None = None, **kwargs: Any) -> Any:
    """Dispatch ``fn`` to the backend of the given arrays.

    Parameters
    ----------
    fn:
        Name of the operation, e.g. ``"sum"`` or ``"linalg.solve"``.
    like:
        Force a backend instead of inferring it from ``args``.
    """
    register_autograd_alias()
    if like is None and _contains_autograd(args):
        # Autograd table/index arrays are plain numpy arrays; force the
        # dispatch backend when the first argument is one of those tables.
        like = "autograd"
    return ar.do(fn, *args, like=like, **kwargs)


def require_same_backend(*arrays: Any) -> str:
    """Return the common backend of ``arrays`` or raise ``TypeError``."""
    return infer_backend(*arrays)


def to_backend(array: Any, backend: str) -> Any:
    """Convert ``array`` to ``backend`` while preserving autograd state."""
    register_autograd_alias()
    if backend == "numpy":
        import numpy as np

        return np.asarray(array)
    if backend == "torch":
        import numpy as np
        import torch

        if isinstance(array, torch.Tensor):
            return array
        # Go through numpy so Python floats keep float64 instead of torch's
        # float32 default (important for scalar kernel parameters).
        return torch.as_tensor(np.asarray(array))
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(array)
    if backend == "autograd":
        import autograd.numpy as anp

        return anp.asarray(array)
    raise ValueError(f"Unknown backend: {backend!r}")


def to_numpy(array: Any) -> Any:
    """Convert a backend array to ``numpy`` (detaching any autograd graph)."""
    register_autograd_alias()
    import numpy as np

    if isinstance(array, np.ndarray):
        return array
    backend = ar.infer_backend(array)
    if backend == "torch":
        return array.detach().cpu().numpy()
    if backend == "jax":
        return np.asarray(array)
    if backend == "autograd":
        return np.asarray(getattr(array, "_value", array))
    return np.asarray(array)


def dtype_of(array: Any) -> Any:
    """Return the dtype of ``array`` (untouched, backend object)."""
    dtype = getattr(array, "dtype", None)
    if dtype is None:
        return "float64"
    return dtype
