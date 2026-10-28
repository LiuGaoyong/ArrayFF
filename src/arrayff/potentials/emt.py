"""Effective medium theory potential.

Follows the implementation in :mod:`ase.calculators.emt` (Jacobsen, Stoltze
and Norskov, Surf. Sci. 366, 394 (1996)), rewritten as a vectorised kernel over
the padded neighbour table. The energy matches ASE; forces are obtained by
automatic differentiation instead of hand coded derivatives.
"""

from __future__ import annotations

from math import log, sqrt
from typing import Any

import numpy as np
from ase.data import chemical_symbols
from ase.units import Bohr

from arrayff.potentials._pairs import pair_distances
from arrayff.potentials.base import Potential
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import do, to_backend
from arrayff.utils.neighbors import NeighborTable

__all__ = ["EMT", "EMT_PARAMETERS", "beta"]

#: (E0 [eV], s0 [bohr], V0 [eV], eta2 [1/bohr], kappa [1/bohr],
#:  lambda [1/bohr], n0 [1/bohr^3])
EMT_PARAMETERS: dict[
    str, tuple[float, float, float, float, float, float, float]
] = {
    "Al": (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
    "Cu": (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
    "Ag": (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
    "Au": (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
    "Ni": (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
    "Pd": (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
    "Pt": (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
    "H": (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
    "C": (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
    "N": (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
    "O": (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850),
}

#: ``(16 * pi / 3) ** (1/3) / sqrt(2)`` with the historical rounding.
beta = 1.809

#: Parameters at which the logistic cutoff is ``1e-4``.
_CUTOFF_EPS = 1e-4


def _gamma1_gamma2(
    s0: float, eta2: float, kappa: float, rc: float, acut: float
):
    """Return the two fcc lattice sums used by EMT (ASE convention)."""
    n = np.array([12.0, 6.0, 24.0])
    r = beta * s0 * np.sqrt([1.0, 2.0, 3.0])
    w = 1.0 / (1.0 + np.exp(acut * (r - rc)))
    x = n * w / 12.0
    gamma1 = float(x @ np.exp(-eta2 * (r - beta * s0)))
    gamma2 = float(x @ np.exp(-kappa / beta * (r - beta * s0)))
    return gamma1, gamma2


def _cutoff(
    asap_cutoff: bool, numbers: np.ndarray | None
) -> tuple[float, float, float]:
    """Return ``(rc, rc_list)`` of the logistic cutoff."""
    if asap_cutoff and numbers is not None:
        symbols = {chemical_symbols[int(z)] for z in np.unique(numbers)}
        s0_max = max(EMT_PARAMETERS[s][1] for s in symbols) * Bohr
    else:
        s0_max = max(p[1] for p in EMT_PARAMETERS.values()) * Bohr
    r1nn = beta * s0_max
    rc = r1nn * 0.5 * (sqrt(3.0) + 2.0)
    r4nn = 2.0 * r1nn
    acut = log(1.0 / _CUTOFF_EPS - 1.0) / (r4nn - rc)
    rc_list = rc * 1.045 if asap_cutoff else rc + 0.5
    return rc, rc_list, acut


@register_potential("emt")
class EMT(Potential):
    """Effective medium theory for Al, Cu, Ag, Au, Ni, Pd, Pt (plus H/C/N/O)."""

    name = "emt"

    def __init__(
        self,
        cutoff: float | None = None,
        *,
        asap_cutoff: bool = False,
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
        builder: str = "sklearn",
    ) -> None:
        self.asap_cutoff = bool(asap_cutoff)
        self.rc, rc_list, self.acut = _cutoff(self.asap_cutoff, None)
        super().__init__(
            cutoff or rc_list,
            backend=backend,
            dtype=dtype,
            max_neighbors=max_neighbors,
            builder=builder,
        )

    def _cutoff_for(self, atoms) -> float:
        if not self.asap_cutoff:
            return self.cutoff
        return _cutoff(True, np.asarray(atoms.numbers))[1]

    def element_parameters(self, elements: np.ndarray) -> dict[str, Any]:
        rows = []
        for z in elements:
            symbol = chemical_symbols[int(z)]
            if symbol not in EMT_PARAMETERS:
                raise NotImplementedError(
                    f"No EMT potential for {symbol} (Z={int(z)})"
                )
            rows.append(EMT_PARAMETERS[symbol])
        table = np.array(rows, dtype=float)
        s0 = table[:, 1] * Bohr
        eta2 = table[:, 3] / Bohr
        kappa = table[:, 4] / Bohr
        gamma1 = np.empty(len(rows))
        gamma2 = np.empty(len(rows))
        for i, (s0_i, eta2_i, kappa_i) in enumerate(zip(s0, eta2, kappa)):
            gamma1[i], gamma2[i] = _gamma1_gamma2(
                float(s0_i), float(eta2_i), float(kappa_i), self.rc, self.acut
            )
        data = {
            "E0": table[:, 0],
            "s0": s0,
            "V0": table[:, 2],
            "eta2": eta2,
            "kappa": kappa,
            "lambda": table[:, 5] / Bohr,
            "n0": table[:, 6] / Bohr**3,
            "inv12gamma1": 1.0 / (12.0 * gamma1),
            "neghalfv0overgamma2": -0.5 * table[:, 2] / gamma2,
        }
        return {
            key: to_backend(value, self.backend) for key, value in data.items()
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
            # ASE convention: an atom without neighbours contributes -E0.
            return -do("sum", params["E0"][species])

        index = table.index
        mask = table.mask
        species_j = species[index]

        # Centre and neighbour parameters, broadcast to (N, 1) and (N, M).
        s0_i = params["s0"][species][:, None]
        eta2_i = params["eta2"][species][:, None]
        kappa_i = params["kappa"][species][:, None]
        lam_i = params["lambda"][species][:, None]
        e0_i = params["E0"][species][:, None]
        v0_i = params["V0"][species][:, None]

        s0_j = params["s0"][species_j]
        eta2_j = params["eta2"][species_j]
        kappa_j = params["kappa"][species_j]
        n0_i = params["n0"][species][:, None]
        n0_j = params["n0"][species_j]
        chi = n0_j / n0_i

        distance = pair_distances(positions, table, cell)
        distance = do("where", mask, distance, 1.0)

        weight = 1.0 / (1.0 + do("exp", self.acut * (distance - self.rc)))
        weight = do("where", mask, weight, 0.0)

        sigma1 = do(
            "sum",
            do("exp", -eta2_j * (distance - beta * s0_j)) * chi * weight,
            axis=-1,
        )
        has_neighbor = do("any", mask, axis=-1)

        inv12gamma1 = params["inv12gamma1"][species]
        sigma1_safe = do("where", has_neighbor, sigma1, 1.0)
        ds = -do("log", sigma1_safe * inv12gamma1) / (beta * eta2_i[:, 0])
        ds = ds[:, None]

        e_embedding = e0_i * (1.0 + lam_i * ds) * do(
            "exp", -lam_i * ds
        ) + 6.0 * v0_i * do("exp", -kappa_i * ds)
        nhalf_i = params["neghalfv0overgamma2"][species][:, None]
        nhalf_j = params["neghalfv0overgamma2"][species_j]
        e_pair = 0.5 * do(
            "sum",
            nhalf_i
            * do("exp", -kappa_j * (distance / beta - s0_j))
            * chi
            * weight
            + nhalf_j
            * do("exp", -kappa_i * (distance / beta - s0_i))
            / chi
            * weight,
            axis=-1,
        )

        energy = do(
            "where",
            has_neighbor,
            e_embedding[:, 0] + e_pair - e0_i[:, 0],
            -e0_i[:, 0],
        )
        return do("sum", energy)
