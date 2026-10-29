"""Base class shared by all potentials.

A potential is defined by a single *array level* function
``energy_arrays(positions, species, table, cell, pbc, params)`` written with
:func:`arrayff.utils.dispatch.do`. The class then provides the
:class:`ase.Atoms` level API, the conversion between the two and the forces
via :mod:`arrayff.ad`.
"""

from __future__ import annotations

import abc
import hashlib
from typing import Any

import numpy as np
from ase import Atoms

from arrayff import ad
from arrayff.neighbors._table import NeighborEdgeList, NeighborPaddedTable
from arrayff.utils.ase import atoms_to_arrays, species_of
from arrayff.utils.dispatch import to_backend, to_numpy

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
    builder:
        Neighbour list implementation: ``"sklearn"`` (default), ``"vesin"``,
        ``"matscipy"``, ``"pymatgen"`` or ``"ase_kdtree"``.
    """

    name: str = "potential"

    def __init__(
        self,
        cutoff: float,
        *,
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
        builder: str = "sklearn",
        skin: float = 0.0,
    ) -> None:
        self.cutoff = float(cutoff)
        self.backend = backend
        self.dtype = dtype
        self.max_neighbors = max_neighbors
        self.builder = builder
        self.skin = float(skin)
        if self.skin < 0.0:
            raise ValueError("skin must be non-negative")
        self._table_static: tuple | None = None
        self._table_positions: np.ndarray | None = None
        self._table_cache: NeighborPaddedTable | None = None

    # -- to be implemented by subclasses ---------------------------------------
    @abc.abstractmethod
    def element_parameters(self, elements: np.ndarray) -> dict[str, Any]:
        """Return backend arrays of the parameters for ``elements``."""

    @abc.abstractmethod
    def energy_arrays(
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable,
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
        table = self._table(atoms)
        return (
            positions,
            species,
            table,
            cell,
            pbc,
            self.element_parameters(elements),
        )

    def _table_static_key(self, atoms: Atoms) -> tuple:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray(atoms.cell.array, dtype=np.float64).tobytes())
        digest.update(np.asarray(atoms.pbc, dtype=bool).tobytes())
        digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
        return (
            digest.hexdigest(),
            self._cutoff_for(atoms),
            self.max_neighbors,
            self.builder,
            self.backend,
            self.skin,
        )

    def _table(self, atoms: Atoms) -> NeighborPaddedTable | NeighborEdgeList:
        """Return the neighbour table, cached according to ``skin``.

        * ``skin == 0``: reuse only when the geometry is bitwise unchanged.
        * ``skin > 0``: build with ``cutoff + skin`` and reuse while no atom
          moved more than ``skin / 2`` (the standard rebuild criterion).
        """
        static = self._table_static_key(atoms)
        positions = np.asarray(atoms.positions, dtype=np.float64)
        if (
            self._table_cache is not None
            and self._table_static == static
            and self._table_positions is not None
        ):
            displacement = float(
                np.linalg.norm(positions - self._table_positions, axis=1).max(
                    initial=0.0
                )
            )
            tolerance = self.skin / 2.0 if self.skin > 0.0 else 0.0
            if displacement <= tolerance:
                return self._table_cache

        table = self._build_table(atoms, self._cutoff_for(atoms) + self.skin)
        self._table_static = static
        self._table_positions = positions.copy()
        self._table_cache = table
        return table

    def stress(self, atoms: Atoms, step: float = 1e-5) -> np.ndarray:
        """Return the 3x3 stress tensor in eV/Angstrom^3.

        Computed by central finite differences of the energy under affine
        strain ``r -> r (1 + eps)``, ``cell -> cell (1 + eps)``. This is
        backend agnostic and costs twelve energy evaluations.
        """
        if atoms.cell.rank != 3:
            raise ValueError("stress requires a fully periodic cell")
        volume = float(atoms.get_volume())
        positions = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        identity = np.eye(3)
        voigt = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        derivative = np.zeros(6)
        for index, (a, b) in enumerate(voigt):
            for sign in (1.0, -1.0):
                strain = np.zeros((3, 3))
                strain[a, b] = sign * step
                if a != b:
                    strain[b, a] = sign * step
                moved = atoms.copy()
                moved.set_cell(cell @ (identity + strain), scale_atoms=False)
                moved.positions = positions @ (identity + strain)
                self.clear_cache()
                energy = self.energy(moved)
                derivative[index] += sign * energy / (2.0 * step)
            if a != b:
                derivative[index] *= 0.5
        sigma = derivative / volume
        return np.array(
            [
                [sigma[0], sigma[5], sigma[4]],
                [sigma[5], sigma[1], sigma[3]],
                [sigma[4], sigma[3], sigma[2]],
            ]
        )

    def _build_table(self, atoms: Atoms, cutoff: float) -> NeighborPaddedTable:
        """Build the neighbour table; pairwise potentials override this."""
        return NeighborPaddedTable.from_ase(
            atoms,
            cutoff,
            max_neighbors=self.max_neighbors,
            backend=self.backend,
            builder=self.builder,
        )

    def clear_cache(self) -> None:
        """Drop the cached neighbour table."""
        self._table_cache = None
        self._table_positions = None
        self._table_static = None

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
