"""Vectorised pair geometry shared by the short ranged potentials."""

from __future__ import annotations

from typing import Any

from arrayff.utils.dispatch import do, dtype_of
from arrayff.utils.neighbors import NeighborTable

__all__ = ["pair_distances", "pair_vectors"]


def pair_vectors(positions: Any, table: NeighborTable, cell: Any) -> Any:
    """Return the ``(N, M, 3)`` pair vectors ``r_j - r_i + S @ cell``.

    The shift term is skipped when ``cell`` is ``None``.
    """
    vectors = positions[table.index] - positions[:, None, :]
    if cell is not None:
        shift = do("astype", table.shift, dtype_of(positions))
        vectors = vectors + do("einsum", "nmk,kl->nml", shift, cell)
    return vectors


def pair_distances(positions: Any, table: NeighborTable, cell: Any) -> Any:
    """Return the ``(N, M)`` pair distances (padding entries are bogus)."""
    vectors = pair_vectors(positions, table, cell)
    return do("sqrt", do("sum", vectors * vectors, axis=-1))
