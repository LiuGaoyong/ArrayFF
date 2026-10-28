"""Backend agnostic autodiff transforms.

The public API mirrors JAX/JAX-like conventions so that user code does not
change across backends::

    grad(f)(x)                    # d f / d x
    value_and_grad(f)(x)          # (f(x), d f / d x)
    jacrev(f)(x)                  # Jacobian
    jvp(f)( (x,), (v,) )          # (f(x), J v)
    vjp(f)( (x,) )                # (f(x), pullback)

Transforms are resolved through autoray's function registry
(:mod:`arrayff.ad.registry`), so adding a backend means registering its
factories, not touching this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import autoray as ar

from arrayff.ad.registry import register_backend
from arrayff.utils.backend import resolve_ad_backend

__all__ = ["grad", "jacrev", "jvp", "value_and_grad", "vjp"]


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
