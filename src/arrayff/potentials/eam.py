"""Embedded atom method (EAM) potential.

Follows ``ase.calculators.eam``::

    E = sum_i F_i(rho_i) + 1/2 sum_{i,j} phi_{a_i,a_j}(r_ij)
    rho_i = sum_{j != i} rho_{a_j}(r_ij)

All three functions are tabulated on uniform grids and evaluated through
:func:`arrayff.ops.interp.interp_rows`, so the kernel stays vectorised and
differentiable on every backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.io.eam import EAMTable, read_setfl
from arrayff.ops.interp import interp_rows
from arrayff.potentials._pairs import pair_distances
from arrayff.potentials.base import Potential
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import do, to_backend
from arrayff.utils.neighbors import NeighborTable

__all__ = ["EAM"]


@register_potential("eam")
class EAM(Potential):
    """Tabulated EAM potential read from an ``setfl`` file."""

    name = "eam"

    def __init__(
        self,
        potential: str,
        cutoff: float | None = None,
        *,
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
        builder: str = "sklearn",
        skin: float = 0.0,
    ) -> None:
        self.table: EAMTable = read_setfl(potential)
        super().__init__(
            cutoff or self.table.cutoff,
            backend=backend,
            dtype=dtype,
            max_neighbors=max_neighbors,
            builder=builder,
            skin=skin,
        )

    def element_parameters(self, elements: np.ndarray) -> dict[str, Any]:
        index = np.array([self.table.index_of(int(z)) for z in elements])
        size = len(elements)
        rphi = self.table.rphi[np.ix_(index, index)].reshape(size * size, -1)
        return {
            "size": size,
            "embedded": to_backend(self.table.embedded[index], self.backend),
            "density": to_backend(self.table.density[index], self.backend),
            "rphi": to_backend(rphi, self.backend),
            "nr": self.table.nr,
            "dr": self.table.dr,
            "nrho": self.table.nrho,
            "drho": self.table.drho,
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

        index = table.index
        mask = table.mask
        size = params["size"]

        species_j = species[index]
        centre = species[:, None]
        distance = pair_distances(positions, table, cell)
        distance = do("where", mask, distance, 1.0)

        # phi_a,b(r) = (r * phi)(r) / r
        inside = do("logical_and", mask, distance < self.cutoff)
        pair_rows = centre * size + species_j
        rphi = interp_rows(
            params["rphi"], pair_rows, distance, 0.0, params["dr"]
        )
        pair_energy = do("sum", do("where", inside, rphi / distance, 0.0))
        pair_energy = 0.5 * pair_energy

        # rho_i = sum_j rho_{a_j}(r_ij)
        density = interp_rows(
            params["density"], species_j, distance, 0.0, params["dr"]
        )
        density = do("sum", do("where", inside, density, 0.0), axis=-1)

        embedding = interp_rows(
            params["embedded"], species, density, 0.0, params["drho"]
        )
        return pair_energy + do("sum", embedding)
