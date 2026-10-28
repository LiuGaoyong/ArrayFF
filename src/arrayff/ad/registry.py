"""Register automatic differentiation transforms with :mod:`autoray`.

Autoray already dispatches array operations, but it has no notion of ``grad``
or ``jacrev``. We register one *factory* per backend under a common name, so
that the rest of arrayff can look transforms up exactly like any other
operation::

    factory = ar.get_lib_fn(backend, "grad")
    transformed = factory(fn)

Each registered entry maps ``f -> transformed`` and accepts a single array
positional argument. Backend specific calling conventions are normalised here
(not in the kernels).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import autoray as ar

from arrayff.utils.backend import is_available

__all__ = ["registered_backends", "register_backend"]

_REGISTERED: set[str] = set()

#: Transform names exposed by arrayff and registered per backend.
TRANSFORM_NAMES = ("grad", "value_and_grad", "jacrev", "jvp", "vjp")


def _jvp_from_immediate(module: Any) -> Callable[[Callable], Callable]:
    """Adapt ``module.jvp(f, primals, tangents)`` to ``f -> callable``."""

    def factory(fn: Callable) -> Callable:
        def transformed(primals: tuple, tangents: tuple) -> tuple:
            return module.jvp(fn, primals, tangents)

        return transformed

    return factory


def _vjp_from_immediate(module: Any) -> Callable[[Callable], Callable]:
    """Adapt ``module.vjp(f, primals)`` to ``f -> callable``."""

    def factory(fn: Callable) -> Callable:
        def transformed(primals: tuple) -> tuple:
            return module.vjp(fn, *primals)

        return transformed

    return factory


def _register_jax() -> None:
    import jax

    ar.register_function("jax", "grad", jax.grad)
    ar.register_function("jax", "value_and_grad", jax.value_and_grad)
    ar.register_function("jax", "jacrev", jax.jacrev)
    ar.register_function("jax", "jvp", _jvp_from_immediate(jax))
    ar.register_function("jax", "vjp", _vjp_from_immediate(jax))


def _register_torch() -> None:
    import torch.func as tf

    def value_and_grad(fn: Callable) -> Callable:
        # torch returns (grad, value); arrayff normalises to (value, grad).
        def transformed(*primals: Any) -> tuple:
            grad, value = tf.grad_and_value(fn)(*primals)
            return value, grad

        return transformed

    ar.register_function("torch", "grad", tf.grad)
    ar.register_function("torch", "value_and_grad", value_and_grad)
    ar.register_function("torch", "jacrev", tf.jacrev)
    ar.register_function("torch", "jvp", _jvp_from_immediate(tf))
    ar.register_function("torch", "vjp", _vjp_from_immediate(tf))


def _register_advect() -> None:
    import advect

    def jvp(fn: Callable) -> Callable:
        transformed = advect.jvp(fn)

        def apply(primals: tuple, tangents: tuple) -> tuple:
            return transformed(*primals, tangents=tangents[0])

        return apply

    def vjp(fn: Callable) -> Callable:
        transformed = advect.vjp(fn)

        def apply(primals: tuple) -> tuple:
            return transformed(*primals)

        return apply

    ar.register_function("advect", "grad", advect.grad)
    ar.register_function("advect", "value_and_grad", advect.value_and_grad)
    ar.register_function("advect", "jacrev", advect.jacobian)
    ar.register_function("advect", "jvp", jvp)
    ar.register_function("advect", "vjp", vjp)
    ar.register_function("advect", "stop_gradient", advect.stop_gradient)


def _register_numpy() -> None:
    import numpy as np

    # Plain numpy arrays carry no derivative information: detaching is a no-op.
    ar.register_function("numpy", "stop_gradient", lambda x: np.asarray(x))


def _REGISTRARS() -> dict[str, Callable[[], None]]:
    return {
        "jax": _register_jax,
        "torch": _register_torch,
        "advect": _register_advect,
        "numpy": _register_numpy,
    }


def register_backend(backend: str) -> None:
    """Register the AD transforms of ``backend`` (idempotent, lazy).

    Importing arrayff must not import torch or jax, so registration happens
    the first time a transform is requested for a given backend.
    """
    if backend in _REGISTERED:
        return
    registrar = _REGISTRARS().get(backend)
    if registrar is None:
        raise ValueError(f"No autodiff backend named {backend!r}")
    if backend != "numpy" and not is_available(backend):
        raise ImportError(
            f"The {backend!r} backend is not installed; "
            f"install arrayff[{backend}]"
        )
    registrar()
    _REGISTERED.add(backend)


def registered_backends() -> tuple[str, ...]:
    """Return the backends whose transforms have been registered so far."""
    return tuple(sorted(_REGISTERED))
