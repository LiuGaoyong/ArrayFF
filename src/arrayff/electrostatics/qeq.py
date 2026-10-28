"""Charge equilibration (QEq).

Implements the Rappe-Goddard charge equilibration model with the algebraic
shielded Coulomb kernel ``1 / sqrt(r^2 + lambda^2)`` and the extended matrix
formulation used by DMFF's QEq solver::

    [ H  1 ] [q   ]   [ -chi ]
    [ 1T 0 ] [lam ] = [  Q   ]

``H_ij = k_e / sqrt(r_ij^2 + lambda^2)`` for ``i != j`` and ``H_ii = J_i``.
The charges are solved once and detached before the forces are taken, which is
the Hellmann-Feynman style truncation used by DMFF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from arrayff import ad
from arrayff.utils.ase import atoms_to_arrays, species_of
from arrayff.utils.dispatch import do, to_backend, to_numpy

__all__ = ["QEq"]

#: Coulomb constant in eV * Angstrom / e^2.
KE = 14.399645


class QEq:
    """Charge equilibration for a set of electronegativity/hardness parameters.

    Parameters
    ----------
    chi, J:
        Electronegativity (eV) and idempotential (eV/e^2), either a scalar or
        a mapping ``{atomic_number: value}``.
    shielding:
        Shielding length ``lambda`` in Angstrom.
    q_total:
        Total charge constraint.
    """

    def __init__(
        self,
        chi: float | dict[int, float],
        J: float | dict[int, float],
        shielding: float = 1.0,
        q_total: float = 0.0,
        *,
        backend: str = "numpy",
        dtype: Any = None,
    ) -> None:
        self.chi = chi
        self.J = J
        self.shielding = float(shielding)
        self.q_total = float(q_total)
        self.backend = backend
        self.dtype = dtype

    def _per_element(self, value: float | dict[int, float], elements):
        if isinstance(value, dict):
            return np.array([float(value[int(z)]) for z in elements])
        return np.full(len(elements), float(value))

    def _setup(self, atoms: Atoms):
        elements, species_np = species_of(atoms.numbers)
        positions, _, cell, pbc = atoms_to_arrays(
            atoms, self.backend, self.dtype
        )
        species = to_backend(species_np, self.backend)
        chi = self._per_element(self.chi, elements)[species_np]
        hardness = self._per_element(self.J, elements)[species_np]
        chi = to_backend(chi, self.backend)
        hardness = to_backend(hardness, self.backend)
        return positions, species, cell, pbc, chi, hardness

    def _distances(self, positions: Any, cell: Any, pbc: Any) -> Any:
        vectors = positions[:, None, :] - positions[None, :, :]
        periodic = bool(to_numpy(pbc).any()) if pbc is not None else False
        if cell is not None and periodic:
            inverse = do("linalg.inv", cell)
            fractional = do("einsum", "ijk,kl->ijl", vectors, inverse)
            fractional = fractional - do("round", fractional)
            vectors = do("einsum", "ijk,kl->ijl", fractional, cell)
        squared = do("sum", vectors * vectors, axis=-1)
        return do("sqrt", squared + self.shielding**2)

    def _system(self, positions: Any, chi: Any, hardness: Any, cell, pbc):
        n = positions.shape[0]
        distance = self._distances(positions, cell, pbc)
        eye = do("eye", n, n, like=self.backend)
        kernel = KE / distance * (1.0 - eye)
        matrix = kernel + do("diag", hardness)
        top = do(
            "concatenate",
            [matrix, do("ones", (n, 1), like=self.backend)],
            axis=1,
        )
        bottom = do(
            "concatenate",
            [
                do("ones", (1, n), like=self.backend),
                do("zeros", (1, 1), like=self.backend),
            ],
            axis=1,
        )
        augmented = do("concatenate", [top, bottom], axis=0)
        rhs = do(
            "concatenate",
            [-chi, do("reshape", to_backend(self.q_total, self.backend), (1,))],
            axis=0,
        )
        return augmented, rhs, kernel, distance

    def charges(self, atoms: Atoms) -> np.ndarray:
        """Return the equilibrium charges in units of the elementary charge."""
        positions, _, cell, pbc, chi, hardness = self._setup(atoms)
        augmented, rhs, _, _ = self._system(positions, chi, hardness, cell, pbc)
        solution = do("linalg.solve", augmented, rhs)
        return to_numpy(solution[: positions.shape[0]])

    def energy(self, atoms: Atoms) -> float:
        energy, _ = self.energy_and_forces(atoms)
        return energy

    def _energy(
        self, positions: Any, charges: Any, chi: Any, hardness: Any, cell, pbc
    ) -> Any:
        n = positions.shape[0]
        distance = self._distances(positions, cell, pbc)
        eye = do("eye", n, n, like=self.backend)
        kernel = KE / distance * (1.0 - eye)
        quadratic = do("sum", (charges[:, None] * kernel) * charges[None, :])
        onsite = do("sum", hardness * charges * charges)
        return do("sum", chi * charges) + 0.5 * (quadratic + onsite)

    def energy_and_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        positions, species, cell, pbc, chi, hardness = self._setup(atoms)
        augmented, rhs, _, _ = self._system(positions, chi, hardness, cell, pbc)
        n = positions.shape[0]
        charges = ad.stop_gradient(do("linalg.solve", augmented, rhs)[:n])

        def value(p: Any) -> Any:
            return self._energy(p, charges, chi, hardness, cell, pbc)

        energy, gradient = ad.value_and_grad(value)(positions)
        return float(to_numpy(energy)), -to_numpy(gradient)
