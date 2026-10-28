"""Conversion helpers between :class:`ase.Atoms` and backend arrays.

Arrayff uses :class:`~ase.Atoms` as its structural input/output type. This
module is the only place where an ``Atoms`` object is unpacked into arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from arrayff.utils.dispatch import to_backend

__all__ = ["atoms_to_arrays", "element_numbers", "species_of"]


def atoms_to_arrays(
    atoms: Atoms,
    backend: str = "numpy",
    dtype: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """Return ``positions, numbers, cell, pbc`` arrays for ``atoms``.

    ``positions`` is ``(N, 3)`` float, ``numbers`` is ``(N,)`` int, ``cell`` is
    ``(3, 3)`` float (zeros when no cell is defined) and ``pbc`` is ``(3,)``
    bool.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(
            f"Expected an ase.Atoms instance, got {type(atoms).__name__}"
        )
    positions = np.asarray(atoms.positions, dtype=dtype or np.float64)
    numbers = np.asarray(atoms.numbers, dtype=np.int64)
    cell = np.asarray(atoms.cell.array, dtype=dtype or np.float64)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    return (
        to_backend(positions, backend),
        to_backend(numbers, backend),
        to_backend(cell, backend),
        to_backend(pbc, backend),
    )


def element_numbers(numbers: np.ndarray) -> np.ndarray:
    """Return the sorted unique atomic numbers present in ``numbers``."""
    return np.unique(np.asarray(numbers, dtype=np.int64))


def species_of(numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map atomic numbers to a compact ``0..S-1`` species index.

    Returns ``(elements, species)`` where ``elements`` is the sorted array of
    unique atomic numbers and ``species`` has the same shape as ``numbers``.
    """
    numbers = np.asarray(numbers, dtype=np.int64)
    elements, species = np.unique(numbers, return_inverse=True)
    return elements, species.astype(np.int64)
