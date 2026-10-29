"""Differentiable table lookup.

EAM evaluates three tabulated functions (embedding, density, pair) on a
uniform radial grid.  NumPy/autograd, torch and JAX all differentiate the
piecewise-linear lookup through plain advanced indexing; ``floor`` and
``astype`` contribute zero derivative, so the local segment slope is the
correct gradient.  No backend-specific primitive is needed.

``interp_rows_with_slope`` returns the value and the analytic slope of the
piecewise-linear interpolant, for the EAM analytic force path (Q5=A).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.utils.backend import infer_backend
from arrayff.utils.dispatch import do, to_backend

__all__ = ["interp_rows", "interp_rows_with_slope"]

def _prepare(table: np.ndarray, backend: str) -> np.ndarray:
    if backend == "numpy":
        return table
    return to_backend(table, backend)


def _table_index(x: Any, table: np.ndarray, x0: float, dx: float):
    u = (x - x0) / dx
    last = table.shape[1] - 2
    k = do("clip", do("astype", do("floor", u), "int32"), 0, last)
    f = u - k
    return k, f


def interp_rows(
    table: np.ndarray,
    rows: Any,
    x: Any,
    x0: float,
    dx: float,
) -> Any:
    """Linearly interpolate ``table[rows]`` at ``x``.

    ``table`` has shape ``(R, T)`` on a uniform grid ``r = x0 + i * dx``.
    ``rows`` selects one row per element of ``x`` (same shape as ``x``).
    """
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("interp_rows expects a (R, T>=2) table")
    backend = infer_backend(x)
    table = _prepare(table, backend)
    k, f = _table_index(x, table, x0, dx)
    low = table[rows, k]
    high = table[rows, k + 1]
    return low * (1.0 - f) + high * f


def interp_rows_with_slope(
    table: np.ndarray,
    rows: Any,
    x: Any,
    x0: float,
    dx: float,
) -> tuple[Any, Any]:
    """Return ``(value, d value / d x)`` for the linear interpolant."""
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("interp_rows_with_slope expects a (R, T>=2) table")
    backend = infer_backend(x)
    table = _prepare(table, backend)
    k, f = _table_index(x, table, x0, dx)
    low = table[rows, k]
    high = table[rows, k + 1]
    value = low * (1.0 - f) + high * f
    slope = (high - low) / dx
    return value, slope
