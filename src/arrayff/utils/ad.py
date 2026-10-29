"""Backend agnostic automatic differentiation.

This module merges the former ``arrayff.ad`` package into a single file:
detach helpers, the autoray transform registry and the public transforms.
Importing it never imports torch/jax/advect; registration happens when a
transform is first requested for a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import autoray as ar

from arrayff.utils.backend import is_available, resolve_ad_backend
from arrayff.utils.dispatch import do

__all__ = [
    "grad",
    "jacrev",
    "jvp",
    "register_backend",
    "registered_backends",
    "stop_gradient",
    "value_and_grad",
    "vjp",
]


# ---------------------------------------------------------------------------
# detach
# ---------------------------------------------------------------------------
def stop_gradient(x: Any) -> Any:
    """Return ``x`` with its gradient contribution removed.

    Dispatches through autoray, so the same call works for numpy (identity),
    torch (``.detach()``), JAX (``lax.stop_gradient``) and advect.
    """
    return do("stop_gradient", x)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
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
    """Register the AD transforms of ``backend`` (idempotent, lazy)."""
    if backend in _REGISTERED:
        return
    registrar = _REGISTRARS().get(backend)
    if registrar is None:
        raise ValueError(f"No autodiff backend named {backend!r}")
    if backend != "numpy" and not is_available(backend):
        raise ImportError(
            f"The {backend!r} backend is not installed; install its extra"
        )
    registrar()
    _REGISTERED.add(backend)


def registered_backends() -> tuple[str, ...]:
    """Return the backends whose transforms have been registered so far."""
    return tuple(sorted(_REGISTERED))


# ---------------------------------------------------------------------------
# public transforms
# ---------------------------------------------------------------------------
def _unary(f: Callable[..., Any], argnum: int, primals: tuple) -> Callable:
    """Freeze every argument of ``f`` except ``primals[argnum]``."""

    def g(x: Any) -> Any:
        args = list(primals)
        args[argnum] = x
        return f(*args)

    return g


def _factory(name: str, primals: tuple, argnum: int) -> Callable:
    backend = resolve_ad_backend(primals[argnum])
    register_backend(backend)
    return ar.get_lib_fn(backend, name)


def grad(f: Callable[..., Any], argnum: int = 0) -> Callable[..., Any]:
    """Return a function computing ``d f / d primals[argnum]``."""

    def wrapped(*primals: Any) -> Any:
        primals = tuple(primals)
        factory = _factory("grad", primals, argnum)
        return factory(_unary(f, argnum, primals))(primals[argnum])

    return wrapped


def value_and_grad(
    f: Callable[..., Any], argnum: int = 0
) -> Callable[..., Any]:
    """Return ``(value, gradient)`` when called."""

    def wrapped(*primals: Any) -> tuple[Any, Any]:
        primals = tuple(primals)
        factory = _factory("value_and_grad", primals, argnum)
        return factory(_unary(f, argnum, primals))(primals[argnum])

    return wrapped


def jacrev(f: Callable[..., Any], argnum: int = 0) -> Callable[..., Any]:
    """Return a function computing the Jacobian of ``f``."""

    def wrapped(*primals: Any) -> Any:
        primals = tuple(primals)
        factory = _factory("jacrev", primals, argnum)
        return factory(_unary(f, argnum, primals))(primals[argnum])

    return wrapped


def _as_tuple(value: Any) -> tuple:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


def jvp(f: Callable[..., Any], argnum: int = 0) -> Callable[..., Any]:
    """Return ``f -> (primals, tangents) -> (value, jvp)``."""

    def wrapped(primals: Any, tangents: Any) -> tuple[Any, Any]:
        primals = _as_tuple(primals)
        tangents = _as_tuple(tangents)
        factory = _factory("jvp", primals, argnum)
        anchor = primals[argnum]
        tangents = (tangents[argnum] if len(tangents) > 1 else tangents[0],)
        return factory(_unary(f, argnum, primals))((anchor,), tangents)

    return wrapped


def vjp(f: Callable[..., Any], argnum: int = 0) -> Callable[..., Any]:
    """Return ``f -> primals -> (value, pullback)``."""

    def wrapped(primals: Any) -> tuple[Any, Callable]:
        primals = _as_tuple(primals)
        factory = _factory("vjp", primals, argnum)
        value, pullback = factory(_unary(f, argnum, primals))(
            (primals[argnum],)
        )

        def vjp_fn(cotangent: Any) -> tuple:
            result = pullback(cotangent)
            return result if isinstance(result, tuple) else (result,)

        return value, vjp_fn

    return wrapped
