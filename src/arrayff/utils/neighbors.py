"""Neighbour tables in a backend friendly, statically shaped form.

Kernels need fixed shape neighbour arrays so that JAX can trace them, while
:mod:`arrayff.neighborlist` (sklearn backend by default) returns the usual
flat ``(i, j, S)`` edge list. This module converts one into the other with
vectorised numpy operations only: no Python loop over atoms or pairs.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from ase import Atoms

from arrayff.neighborlist import neighbor_list
from arrayff.utils.dispatch import to_backend

__all__ = ["NeighborTable", "neighbor_table"]


class NeighborTable(NamedTuple):
    """Padded neighbour list.

    Attributes
    ----------
    index:
        ``(N, M)`` int array of neighbour indices; padding entries point at
        atom 0 and must be ignored via ``mask``.
    shift:
        ``(N, M, 3)`` int array of periodic image shifts ``S`` such that
        ``d_ij = r_j - r_i + S @ cell``.
    mask:
        ``(N, M)`` bool array, ``True`` for real neighbours.
    """

    index: np.ndarray
    shift: np.ndarray
    mask: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.index.shape[0])

    @property
    def max_neighbors(self) -> int:
        return int(self.index.shape[1])

    def to_backend(self, backend: str) -> NeighborTable:
        """Return a copy whose arrays live on ``backend``."""
        if backend == "numpy":
            return self
        return NeighborTable(
            index=to_backend(self.index, backend),
            shift=to_backend(self.shift, backend),
            mask=to_backend(self.mask, backend),
        )


def _empty_table(n_atoms: int) -> NeighborTable:
    return NeighborTable(
        index=np.zeros((n_atoms, 0), dtype=np.int64),
        shift=np.zeros((n_atoms, 0, 3), dtype=np.int64),
        mask=np.zeros((n_atoms, 0), dtype=bool),
    )


def neighbor_table(
    atoms: Atoms,
    cutoff: float,
    *,
    max_neighbors: int | None = None,
    backend: str = "numpy",
    builder: str = "sklearn",
) -> NeighborTable:
    """Build a padded neighbour table for ``atoms``.

    Parameters
    ----------
    cutoff:
        Global cutoff radius in Angstrom.
    max_neighbors:
        Fixed second dimension of the table. Required for JAX when the shape
        must stay constant between calls; defaults to the largest neighbour
        count found.
    builder:
        Neighbour list implementation, ``"sklearn"`` (default) or any other
        backend supported by :func:`arrayff.neighborlist.neighbor_list`.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(
            f"Expected an ase.Atoms instance, got {type(atoms).__name__}"
        )
    n_atoms = len(atoms)
    if n_atoms == 0 or cutoff <= 0.0:
        return _empty_table(n_atoms)

    i, j, shift = neighbor_list(
        "ijS",
        a=atoms,
        cutoff=float(cutoff),
        backend=builder,
        bothways=True,
        sorted=False,
    )
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    if len(i) == 0:
        return _empty_table(n_atoms)

    counts = np.bincount(i, minlength=n_atoms)
    width = int(np.max(counts, initial=0))
    if max_neighbors is not None:
        if int(max_neighbors) < width:
            raise ValueError(
                f"max_neighbors={max_neighbors} is smaller than the largest "
                f"neighbour count ({width})."
            )
        width = int(max_neighbors)

    # Vectorised grouping: sort by centre atom, then compute each entry's
    # column inside its row from the running offsets.
    order = np.argsort(i, kind="stable")
    rows = i[order]
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    cols = np.arange(len(order), dtype=np.int64) - offsets[rows]

    index = np.zeros((n_atoms, width), dtype=np.int64)
    table_shift = np.zeros((n_atoms, width, 3), dtype=np.int64)
    mask = np.zeros((n_atoms, width), dtype=bool)
    index[rows, cols] = j[order]
    table_shift[rows, cols] = shift[order]
    mask[rows, cols] = True

    table = NeighborTable(index=index, shift=table_shift, mask=mask)
    return table.to_backend(backend)
