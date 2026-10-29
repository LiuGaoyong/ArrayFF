"""Reader for the EAM ``setfl`` (alloy) format.

The layout follows LAMMPS / ``ase.calculators.eam``:

* three header lines,
* ``nelements``, element names, ``nrho drho nr dr cutoff``,
* per element: ``Z mass a lattice`` + ``nrho`` embedding values + ``nr``
  density values,
* ``nr`` values of ``r * phi`` for every ``i >= j`` pair.

Tabulated functions are stored as plain arrays so any backend can evaluate
them; interpolation happens in :mod:`arrayff.ops.interp`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

__all__ = ["EAMTable", "read_setfl"]


@dataclass(frozen=True)
class EAMTable:
    """Tabulated EAM functions on uniform grids."""

    elements: tuple[int, ...]
    z: np.ndarray
    nrho: int
    drho: float
    nr: int
    dr: float
    cutoff: float
    embedded: np.ndarray  # (S, nrho)
    density: np.ndarray  # (S, nr)
    rphi: np.ndarray  # (S, S, nr), symmetric

    def index_of(self, atomic_number: int) -> int:
        matches = np.flatnonzero(self.z == int(atomic_number))
        if len(matches) == 0:
            raise KeyError(
                f"Element Z={atomic_number} is not present in the EAM table "
                f"({list(self.z)})"
            )
        return int(matches[0])


def read_setfl(path: str | pathlib.Path) -> EAMTable:
    """Read an EAM ``setfl`` file."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    data: list[str] = []
    for line in lines[3:]:
        data.extend(line.split())

    nelements = int(data[0])
    cursor = 1
    data[cursor : cursor + nelements]
    cursor += nelements
    nrho = int(data[cursor])
    drho = float(data[cursor + 1])
    nr = int(data[cursor + 2])
    dr = float(data[cursor + 3])
    cutoff = float(data[cursor + 4])
    cursor += 5

    embedded = np.zeros((nelements, nrho))
    density = np.zeros((nelements, nr))
    z = np.zeros(nelements, dtype=int)
    for element in range(nelements):
        z[element] = int(data[cursor])
        cursor += 4  # Z, mass, lattice constant, lattice type name
        embedded[element] = np.asarray(
            data[cursor : cursor + nrho], dtype=float
        )
        cursor += nrho
        density[element] = np.asarray(data[cursor : cursor + nr], dtype=float)
        cursor += nr

    rphi = np.zeros((nelements, nelements, nr))
    for i in range(nelements):
        for j in range(i + 1):
            rphi[j, i] = np.asarray(data[cursor : cursor + nr], dtype=float)
            rphi[i, j] = rphi[j, i]
            cursor += nr

    return EAMTable(
        elements=tuple(int(x) for x in z),
        z=z,
        nrho=nrho,
        drho=drho,
        nr=nr,
        dr=dr,
        cutoff=cutoff,
        embedded=embedded,
        density=density,
        rphi=rphi,
    )
