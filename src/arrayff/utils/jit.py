"""Compilation helpers.

Automatic compilation is opt-in and always applied to *array level*
functions: :class:`ase.Atoms` never enters a compiled function because its
contents are Python objects that JAX cannot trace.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import autoray as ar


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    backend: str | None = None,
    compiler_opts: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Compile ``fn`` with the backend compiler.

    ``backend="jax"`` uses ``jax.jit``; torch uses ``torch.jit.trace`` via
    autoray.  Functions written with :func:`arrayff.utils.dispatch.do` can be
    traced; functions that also call backend AD transforms cannot (except
    JAX).

    Only functions written with :func:`arrayff.utils.dispatch.do` are traced,
    so kernels must not call backend specific functions directly.
    """
    return ar.autojit(fn, backend=backend, compiler_opts=compiler_opts)


def autojit(fn: Callable[..., Any], backend: str) -> Callable[..., Any]:
    """Alias used by :class:`arrayff.potentials.base.Potential`."""
    return ar.autojit(fn, backend=backend)


def enable_x64() -> None:
    """Enable 64 bit floats in JAX (off by default)."""
    import jax

    jax.config.update("jax_enable_x64", True)


def x64_enabled() -> bool:
    """Return whether JAX currently uses 64 bit floats."""
    import jax

    return bool(jax.config.read("jax_enable_x64"))
