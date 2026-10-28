"""Lennard-Jones potential.

Two parameter modes are supported:

``kim`` (default)
    The universal 6-12 parameters of the OpenKIM model
    ``LennardJones612_UniversalShifted__MO_959249795837_003``: sigma from
    covalent radii, epsilon from bond dissociation energies, Lorentz-Berthelot
    mixing and each pair interaction shifted to zero at ``4 * sigma_ij``.

``explicit``
    User supplied ``sigma``/``epsilon`` with a global ``cutoff`` (optionally
    energy shifted at that cutoff).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.potentials._pairs import pair_distances
from arrayff.potentials.base import Potential
from arrayff.potentials.lj_kim import KIM_LJ_PARAMETERS
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import do, to_backend
from arrayff.utils.neighbors import NeighborTable

__all__ = ["LennardJones"]

#: Cutoff of the universal KIM model is four sigma.
KIM_CUTOFF_FACTOR = 4.0


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
    """Lennard-Jones 6-12 potential with Lorentz-Berthelot mixing.

    With no ``sigma``/``epsilon`` given, the OpenKIM universal shifted
    parameters are used. Passing explicit values switches to a global cutoff
    model, matching e.g. ``ase.calculators.lj``.
    """

    name = "lj"

    def __init__(
        self,
        sigma: float | dict[int, float] | None = None,
        epsilon: float | dict[int, float] | None = None,
        cutoff: float | None = None,
        *,
        shift: bool | None = None,
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
    ) -> None:
        self.kim = sigma is None and epsilon is None
        if self.kim:
            if cutoff is not None:
                raise ValueError(
                    "The KIM universal model fixes the cutoff at 4*sigma; "
                    "pass explicit sigma/epsilon to use a global cutoff."
                )
            self.cutoff = KIM_CUTOFF_FACTOR * max(
                sigma_ for _, sigma_ in KIM_LJ_PARAMETERS.values()
            )
        else:
            if sigma is None or epsilon is None:
                raise ValueError("Pass both sigma and epsilon, or neither.")
            if cutoff is None:
                raise ValueError("An explicit cutoff is required.")
            self.cutoff = float(cutoff)
        self.shift = self.kim if shift is None else bool(shift)
        super().__init__(
            self.cutoff,
            backend=backend,
            dtype=dtype,
            max_neighbors=max_neighbors,
        )
        self.sigma = sigma
        self.epsilon = epsilon

    def _cutoff_for(self, atoms) -> float:
        if not self.kim:
            return self.cutoff
        present = np.unique(np.asarray(atoms.numbers, dtype=np.int64))
        return KIM_CUTOFF_FACTOR * max(
            KIM_LJ_PARAMETERS[int(z)][1] for z in present
        )

    def _parameters(self, elements: np.ndarray) -> dict[str, Any]:
        if self.kim:
            try:
                values = [KIM_LJ_PARAMETERS[int(z)] for z in elements]
            except KeyError as exc:
                raise KeyError(
                    f"No KIM Lennard-Jones parameters for Z={exc.args[0]}"
                ) from None
            epsilon = np.array([v[0] for v in values])
            sigma = np.array([v[1] for v in values])
        else:
            sigma = _per_element(self.sigma, elements)  # type: ignore[arg-type]
            epsilon = _per_element(
                self.epsilon,
                elements,  # type: ignore[arg-type]
            )
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
        if table.index.shape[1] == 0:
            return do("sum", positions * 0.0)

        sigma = params["sigma"]
        epsilon = params["epsilon"]
        index = table.index
        mask = table.mask

        species_j = species[index]
        sigma_ij = 0.5 * (sigma[species][:, None] + sigma[species_j])
        epsilon_ij = do("sqrt", epsilon[species][:, None] * epsilon[species_j])

        distance = pair_distances(positions, table, cell)
        distance = do("where", mask, distance, 1.0)

        if self.kim:
            # Per pair cutoff: the model is shifted at 4 * sigma_ij.
            cutoff_ij = KIM_CUTOFF_FACTOR * sigma_ij
            inside = do("logical_and", mask, distance < cutoff_ij)
        else:
            cutoff_ij = self.cutoff
            inside = mask

        sr6 = (sigma_ij / distance) ** 6
        energy = 4.0 * epsilon_ij * (sr6 * sr6 - sr6)
        if self.shift:
            src6 = (sigma_ij / cutoff_ij) ** 6
            energy = energy - 4.0 * epsilon_ij * (src6 * src6 - src6)

        energy = do("where", inside, energy, 0.0)
        return 0.5 * do("sum", energy)
