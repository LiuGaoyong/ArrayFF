"""Differentiable table lookup.

EAM evaluates three tabulated functions (embedding, density, pair) on a
uniform radial grid. NumPy/Advect cannot differentiate through an integer
index derived from a traced value, so for that backend the lookup is wrapped
in a custom primitive with an explicit JVP/transpose (the slope of the linear
segment). Torch and JAX use plain advanced indexing, which they already
differentiate correctly.

``interp_rows_with_slope`` returns the value and the analytic slope of the
piecewise-linear interpolant, for the EAM analytic force path (Q5=A).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.utils.backend import infer_backend
from arrayff.utils.dispatch import do, to_backend

__all__ = ["interp_rows", "interp_rows_with_slope"]

_ADVECT_INTERP = None


def _advect_interp():
    """Build (once) the advect primitive for linear table interpolation."""
    global _ADVECT_INTERP
    if _ADVECT_INTERP is not None:
        return _ADVECT_INTERP

    import advect

    @advect.primitive(static_argnames=("table", "rows", "x0", "dx"))
    def interp(
        x: Any, *, table: np.ndarray, rows: np.ndarray, x0: float, dx: float
    ):
        u = (x - x0) / dx
        k = np.clip(np.floor(u).astype(np.int64), 0, table.shape[1] - 2)
        f = u - k
        return table[rows, k] * (1.0 - f) + table[rows, k + 1] * f

    def _slope(x, table, rows, x0, dx):
        u = (x - x0) / dx
        k = np.clip(np.floor(u).astype(np.int64), 0, table.shape[1] - 2)
        return (table[rows, k + 1] - table[rows, k]) / dx

    @interp.def_jvp
    def _jvp(output, primals, tangents, table, rows, x0, dx):
        (x,), (tangent,) = primals, tangents
        return _slope(x, table, rows, x0, dx) * tangent

    @interp.def_transpose
    def _transpose(cotangent, primals, output, table, rows, x0, dx):
        (x,) = primals
        return (_slope(x, table, rows, x0, dx) * cotangent,)

    _ADVECT_INTERP = interp
    return interp


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
    if backend == "advect":
        interp = _advect_interp()
        return interp(
            x,
            table=np.ascontiguousarray(table, dtype=float),
            rows=np.ascontiguousarray(np.asarray(rows), dtype=np.int64),
            x0=float(x0),
            dx=float(dx),
        )

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
