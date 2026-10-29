"""Helpers for optional dependencies.

Optional libraries (``torch``, ``jax``, alternative neighbour builders, ...)
must never be imported at module import time.  Use :func:`require` inside the
function that actually needs them so that ``import arrayff`` works with only
the hard dependencies installed.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from arrayff.utils.exceptions import MissingBackendError

__all__ = ["is_available", "require"]


def is_available(module: str) -> bool:
    """Return whether ``module`` is importable without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # broken parent package
        return False


def require(module: str, *, extra: str | None = None) -> ModuleType:
    """Import and return ``module`` or raise :class:`MissingBackendError`."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = (
            f"; install it with `pip install arrayff[{extra}]`" if extra else ""
        )
        raise MissingBackendError(
            f"The optional dependency {module!r} is required{hint}"
        ) from exc
