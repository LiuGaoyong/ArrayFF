"""Lennard-Jones potential."""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.potentials._pairs import pair_distances
from arrayff.potentials.base import Potential
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import do, to_backend
from arrayff.utils.neighbors import NeighborTable

__all__ = ["LennardJones"]


def _per_element(
    value: float | dict[int, float], elements: np.ndarray
) -> np.ndarray:
    """Expand a scalar or ``{Z: value}`` map into an element array."""
    if isinstance(value, dict):
        missing = [int(z) for z in elements if int(z) not in value]
        if missing:
            raise KeyError(f"Missing Lennard-Jones parameters for Z={missing}")
        return np.array([float(value[int(z)]) for z in elements])
    return np.full(len(elements), float(value))


@register_potential("lj")
class LennardJones(Potential):
    """Lennard-Jones potential with Lorentz-Berthelot mixing rules.

    Parameters
    ----------
    sigma, epsilon:
        Either a scalar (all elements) or a mapping ``{atomic_number: value}``
        in Angstrom / eV.
    cutoff:
        Cutoff radius in Angstrom.
    shift:
        Shift the energy so that it is continuous at the cutoff.
    """

    name = "lj"

    def __init__(
        self,
        sigma: float | dict[int, float] = 3.4,
        epsilon: float | dict[int, float] = 0.0104,
        cutoff: float = 8.0,
        *,
        shift: bool = False,
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
    ) -> None:
        super().__init__(
            cutoff, backend=backend, dtype=dtype, max_neighbors=max_neighbors
        )
        self.sigma = sigma
        self.epsilon = epsilon
        self.shift = bool(shift)

    def _parameters(self, elements: np.ndarray) -> dict[str, Any]:
        sigma = _per_element(self.sigma, elements)
        epsilon = _per_element(self.epsilon, elements)
        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be positive")
        if np.any(epsilon < 0.0):
            raise ValueError("epsilon must be non-negative")
        return {
            "sigma": to_backend(sigma, self.backend),
            "epsilon": to_backend(epsilon, self.backend),
        }

    def energy_arrays(
        self,
        positions: Any,
        species: Any,
        table: NeighborTable,
        cell: Any,
        pbc: Any,
        params: dict[str, Any],
    ) -> Any:
        sigma = params["sigma"]
        epsilon = params["epsilon"]
        index = table.index
        mask = table.mask

        sigma_i = sigma[species]
        epsilon_i = epsilon[species]
        species_j = species[index]
        sigma_j = sigma[species_j]
        epsilon_j = epsilon[species_j]

        sigma_ij = 0.5 * (sigma_i[:, None] + sigma_j)
        epsilon_ij = do("sqrt", epsilon_i[:, None] * epsilon_j)

        distance = pair_distances(positions, table, cell)
        distance = do("where", mask, distance, 1.0)
        sr6 = (sigma_ij / distance) ** 6
        energy = 4.0 * epsilon_ij * (sr6 * sr6 - sr6)

        if self.shift:
            src6 = (sigma_ij / self.cutoff) ** 6
            energy = energy - 4.0 * epsilon_ij * (src6 * src6 - src6)

        energy = do("where", mask, energy, 0.0)
        return 0.5 * do("sum", energy)
