"""Array backend detection and policy helpers.

Arrayff never branches on the backend inside numerical kernels: the backend
is inferred from the arrays themselves and ``autoray`` dispatches accordingly.
This module only collects the few places where a policy decision is needed
(e.g. which engine provides autodiff for ``numpy`` arrays).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import autoray as ar

#: Backends whose arrays can flow through the kernels.
ARRAY_BACKENDS = ("numpy", "torch", "jax", "autograd")

#: Backends that provide native automatic differentiation.
AD_BACKENDS = ("torch", "jax", "autograd")

#: Backend used to differentiate plain ``numpy`` inputs.
NUMPY_AD_BACKEND = "autograd"


def infer_backend(*arrays: Any) -> str:
    """Return the single backend shared by ``arrays``.

    ``None`` entries are ignored so that optional arguments can be passed
    straight through. Python scalars (``int``/``float``) are ignored too:
    they are weak values that any backend accepts.
    """
    names: set[str] = set()
    for array in arrays:
        if array is None:
            continue
        name = ar.infer_backend(array)
        if name in ("builtins", "dask"):  # pragma: no cover - defensive
            continue
        names.add(name)
    if not names:
        raise TypeError(
            "Cannot infer an array backend from the given arguments; "
            "pass at least one array."
        )
    if len(names) > 1:
        raise TypeError(
            "Mixed array backends are not supported: "
            f"{sorted(names)}. Convert the inputs to one backend first."
        )
    return names.pop()


def resolve_ad_backend(*primals: Any) -> str:
    """Return the backend used to differentiate the given primals.

    Plain ``numpy`` arrays are differentiated with :mod:`autograd`, which is
    the designated NumPy automatic differentiation engine.
    """
    backend = infer_backend(*primals)
    if backend == "numpy":
        return NUMPY_AD_BACKEND
    return backend


def default_dtype(backend: str) -> Any:
    """Return the default floating point dtype for ``backend``."""
    if backend == "numpy":
        import numpy as np

        return np.dtype("float64")
    if backend == "torch":
        import torch

        return torch.float64
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.float64
    if backend == "autograd":
        import numpy as np

        return np.dtype("float64")
    raise ValueError(f"Unknown backend: {backend!r}")


def is_available(name: str) -> bool:
    """Return whether the optional dependency ``name`` is importable."""
    return importlib.util.find_spec(name) is not None


def available_backends() -> tuple[str, ...]:
    """Return the array backends usable in the current environment."""
    names = ["numpy"]
    if is_available("torch"):
        names.append("torch")
    if is_available("jax"):
        names.append("jax")
    if is_available("autograd"):
        names.append("autograd")
    return tuple(names)
