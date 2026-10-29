"""Embedded atom method (EAM) potential.

Follows ``ase.calculators.eam``::

    E = sum_i F_i(rho_i) + 1/2 sum_{i,j} phi_{a_i,a_j}(r_ij)
    rho_i = sum_{j != i} rho_{a_j}(r_ij)

All three functions are tabulated on uniform grids and evaluated through
:func:`arrayff.utils.interp.interp_rows`, so the kernel stays vectorised and
differentiable on every backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.potentials.eam.io.eam import EAMTable, read_setfl
from arrayff.utils.interp import interp_rows, interp_rows_with_slope
from arrayff.potentials.pairwise._pairs import pair_distances, pair_vectors
from arrayff.potentials.base import Potential
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import do, to_backend
from arrayff.neighbors._table import NeighborPaddedTable

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
        jit: bool = False,
        method: str = "analytic",
    ) -> None:
        self.table: EAMTable = read_setfl(potential)
        self._param_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        super().__init__(
            cutoff or self.table.cutoff,
            backend=backend,
            dtype=dtype,
            max_neighbors=max_neighbors,
            builder=builder,
            skin=skin,
            jit=jit,
            method=method,
        )

    def element_parameters(self, elements: np.ndarray) -> dict[str, Any]:
        key = tuple(int(z) for z in elements)
        cached = self._param_cache.get(key)
        if cached is not None:
            return cached
        index = np.array([self.table.index_of(int(z)) for z in elements])
        size = len(elements)
        rphi = self.table.rphi[np.ix_(index, index)].reshape(size * size, -1)
        result = {
            "size": size,
            "embedded": to_backend(self.table.embedded[index], self.backend),
            "density": to_backend(self.table.density[index], self.backend),
            "rphi": to_backend(rphi, self.backend),
            "nr": self.table.nr,
            "dr": self.table.dr,
            "nrho": self.table.nrho,
            "drho": self.table.drho,
        }
        self._param_cache[key] = result
        return result

    def energy_arrays(
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable,
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


    def _eam_derivative_terms(
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable,
        cell: Any,
        params: dict[str, Any],
    ) -> tuple[Any, Any]:
        if table.index.shape[1] == 0:
            zeros = do("zeros_like", positions)
            return zeros, do("zeros", (3, 3), like=positions)

        index = table.index
        mask = table.mask
        size = params["size"]
        species_j = species[index]
        centre = species[:, None]
        distance = pair_distances(positions, table, cell)
        distance = do("where", mask, distance, 1.0)
        inside = do(
            "logical_and",
            mask,
            do("logical_and", distance < self.cutoff, distance > 0.0),
        )
        r_safe = do("where", inside, distance, self.cutoff)

        pair_rows = centre * size + species_j
        rphi, drphi = interp_rows_with_slope(
            params["rphi"], pair_rows, r_safe, 0.0, params["dr"]
        )
        dphi = drphi / r_safe - rphi / (r_safe * r_safe)

        density_value, density_slope = interp_rows_with_slope(
            params["density"], species_j, r_safe, 0.0, params["dr"]
        )
        density = do("sum", do("where", inside, density_value, 0.0), axis=-1)
        embedded, d_embedded = interp_rows_with_slope(
            params["embedded"], species, density, 0.0, params["drho"]
        )
        _ = embedded
        d_embedded_j = d_embedded[index]
        dE_dr = do("where", inside, dphi, 0.0)
        dE_dr = dE_dr + do(
            "where", inside, d_embedded[:, None] * density_slope, 0.0
        )
        dE_dr = dE_dr + do(
            "where", inside, d_embedded_j * density_slope, 0.0
        )
        coefficient = do("where", inside, dE_dr / r_safe, 0.0)
        vectors = pair_vectors(positions, table, cell)
        forces = do("sum", coefficient[:, :, None] * vectors, axis=1)
        virial = 0.5 * do(
            "einsum", "nm,nmi,nmj->ij", coefficient, vectors, vectors
        )
        volume = do("abs", do("linalg.det", cell))
        stress = virial / volume
        return forces, stress

    def evaluate_arrays(  # type: ignore[override]
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable,
        cell: Any,
        pbc: Any,
        params: dict[str, Any],
        *,
        want: tuple[str, ...] = ("energy", "forces", "stress"),
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "energy" in want:
            out["energy"] = self.energy_arrays(
                positions, species, table, cell, pbc, params
            )
        if "forces" in want or "stress" in want:
            forces, stress = self._eam_derivative_terms(
                positions, species, table, cell, params
            )
            if "forces" in want:
                out["forces"] = forces
            if "stress" in want:
                out["stress"] = stress
        return out

    def forces_arrays(  # type: ignore[override]
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable,
        cell: Any,
        pbc: Any,
        params: dict[str, Any],
    ) -> Any:
        return self.evaluate_arrays(
            positions, species, table, cell, pbc, params, want=("forces",)
        )["forces"]

    def stress_arrays(  # type: ignore[override]
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable,
        cell: Any,
        pbc: Any,
        params: dict[str, Any],
    ) -> Any:
        return self.evaluate_arrays(
            positions, species, table, cell, pbc, params, want=("stress",)
        )["stress"]
