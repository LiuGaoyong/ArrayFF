"""Vectorised pair geometry shared by the short ranged potentials."""

from __future__ import annotations

from typing import Any

from arrayff.utils.dispatch import do, dtype_of
from arrayff.neighbors._table import NeighborPaddedTable

__all__ = ["pair_distances", "pair_vectors"]


def pair_vectors(positions: Any, table: NeighborPaddedTable, cell: Any) -> Any:
    """Return the ``(N, M, 3)`` pair vectors ``r_j - r_i + S @ cell``.

    The shift term is skipped when ``cell`` is ``None``.
    """
    vectors = positions[table.index] - positions[:, None, :]
    if cell is not None:
        shift = do("astype", table.shift, dtype_of(positions))
        vectors = vectors + do("einsum", "nmk,kl->nml", shift, cell)
    return vectors


def pair_distances(
    positions: Any, table: NeighborPaddedTable, cell: Any
) -> Any:
    """Return the ``(N, M)`` pair distances.

    Padding entries are replaced by ``1.0`` *before* the square root so that
    autodiff never evaluates ``1 / sqrt(0)`` (which would give ``0 * inf``).
    """
    vectors = pair_vectors(positions, table, cell)
    squared = do("sum", vectors * vectors, axis=-1)
    squared = do("where", table.mask, squared, 1.0)
    return do("sqrt", squared)
