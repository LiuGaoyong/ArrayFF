"""Base class shared by all potentials.

A potential is defined by a single *array level* function
``energy_arrays(positions, species, table, cell, pbc, params)`` written with
:func:`arrayff.utils.dispatch.do`. The class then provides the
:class:`ase.Atoms` level API, the conversion between the two and the forces
via :mod:`arrayff.ad`.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
from ase import Atoms

from arrayff import ad
from arrayff.utils.ase import atoms_to_arrays, species_of
from arrayff.utils.dispatch import to_backend, to_numpy
from arrayff.utils.neighbors import NeighborTable, neighbor_table

__all__ = ["Potential"]


class Potential(abc.ABC):
    """Energy/force evaluator bound to one array backend.

    Parameters
    ----------
    cutoff:
        Global cutoff radius in Angstrom used to build the neighbour table.
    backend:
        ``"numpy"`` (default), ``"torch"`` or ``"jax"``.
    dtype:
        Floating point dtype used when converting the ``Atoms`` positions.
    max_neighbors:
        Fixed neighbour-table width; required to keep shapes static for JAX.
    """

    name: str = "potential"

    def __init__(
        self,
        cutoff: float,
        *,
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
    ) -> None:
        self.cutoff = float(cutoff)
        self.backend = backend
        self.dtype = dtype
        self.max_neighbors = max_neighbors

    # -- to be implemented by subclasses ---------------------------------------
    @abc.abstractmethod
    def _parameters(self, elements: np.ndarray) -> dict[str, Any]:
        """Return backend arrays of the parameters for ``elements``."""

    @abc.abstractmethod
    def energy_arrays(
        self,
        positions: Any,
        species: Any,
        table: NeighborTable,
        cell: Any,
        pbc: Any,
        params: dict[str, Any],
    ) -> Any:
        """Return the total energy as a scalar array."""

    # -- pipeline -----------------------------------------------------------
    def _cutoff_for(self, atoms: Atoms) -> float:
        """Return the neighbour table cutoff for ``atoms``."""
        return self.cutoff

    def _context(self, atoms: Atoms) -> tuple:
        elements, species_np = species_of(atoms.numbers)
        positions, _, cell, pbc = atoms_to_arrays(
            atoms, self.backend, self.dtype
        )
        species = to_backend(species_np, self.backend)
        table = neighbor_table(
            atoms,
            self._cutoff_for(atoms),
            max_neighbors=self.max_neighbors,
            backend=self.backend,
        )
        return positions, species, table, cell, pbc, self._parameters(elements)

    def energy(self, atoms: Atoms) -> float:
        """Return the total energy in eV."""
        positions, species, table, cell, pbc, params = self._context(atoms)
        value = self.energy_arrays(positions, species, table, cell, pbc, params)
        return float(to_numpy(value))

    def energy_and_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Return ``(energy, forces)`` with forces in eV/Angstrom."""
        positions, species, table, cell, pbc, params = self._context(atoms)

        def value(p: Any) -> Any:
            return self.energy_arrays(p, species, table, cell, pbc, params)

        energy, gradient = ad.value_and_grad(value)(positions)
        return float(to_numpy(energy)), -to_numpy(gradient)
