"""Lennard-Jones potential, expressed through the generic pairwise layer."""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.potentials.pairwise.base import PairwisePotential
from arrayff.potentials.pairwise.forms import GeneralizedLJ
from arrayff.potentials.pairwise.lj_kim import KIM_LJ_PARAMETERS
from arrayff.potentials.pairwise.params import PairParameters
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import to_backend

__all__ = ["KIM_LJ_PARAMETERS", "LennardJones"]

#: ``r0 = 2**(1/6) * sigma`` converts the usual ``(sigma, epsilon)``
#: parameterisation into the ``(r0, depth)`` one used by GeneralizedLJ.
SIGMA_TO_R0 = 2.0 ** (1.0 / 6.0)


def _per_element(
    value: float | dict[int, float], elements: np.ndarray
) -> np.ndarray:
    if isinstance(value, dict):
        missing = [int(z) for z in elements if int(z) not in value]
        if missing:
            raise KeyError(f"Missing Lennard-Jones parameters for Z={missing}")
        return np.array([float(value[int(z)]) for z in elements])
    return np.full(len(elements), float(value))


@register_potential("lj")
class LennardJones(PairwisePotential):
    """Lennard-Jones 6-12 potential.

    With no ``sigma``/``epsilon`` the OpenKIM universal shifted parameters are
    used (per element cutoff, geometric mixing). Passing explicit ``sigma`` /
    ``epsilon`` switches to the ASE convention: global cutoff, arithmetic
    sigma mixing, geometric epsilon mixing.
    """

    name = "lj"
    form = GeneralizedLJ(12, 6)

    def __init__(
        self,
        sigma: float | dict[int, float] | None = None,
        epsilon: float | dict[int, float] | None = None,
        cutoff: float | None = None,
        *,
        shift: bool | None = None,
        smooth: bool = False,
        layout: str = "edges",
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
        builder: str = "sklearn",
        skin: float = 0.0,
    ) -> None:
        self.kim = sigma is None and epsilon is None
        if self.kim:
            if cutoff is not None:
                raise ValueError(
                    "The KIM universal model fixes the cutoff at 4*sigma; "
                    "pass explicit sigma/epsilon to use a global cutoff."
                )
            cutoff = max(c for c, _, _ in KIM_LJ_PARAMETERS.values())
            shift = True if shift is None else shift
            mixing = {
                "r0": "geometric",
                "depth": "geometric",
                "cutoff": "geometric",
            }
        else:
            if sigma is None or epsilon is None:
                raise ValueError("Pass both sigma and epsilon, or neither.")
            if cutoff is None:
                raise ValueError("An explicit cutoff is required.")
            shift = False if shift is None else shift
            mixing = {"r0": "arithmetic", "depth": "geometric"}

        super().__init__(
            cutoff,
            shift=shift,
            smooth=smooth,
            layout=layout,
            backend=backend,
            dtype=dtype,
            max_neighbors=max_neighbors,
            builder=builder,
            skin=skin,
        )
        self.sigma = sigma
        self.epsilon = epsilon
        self.mixing = mixing

    def element_parameters(self, elements: np.ndarray) -> PairParameters:
        if self.kim:
            try:
                rows = [KIM_LJ_PARAMETERS[int(z)] for z in elements]
            except KeyError as exc:
                raise KeyError(
                    f"No KIM Lennard-Jones parameters for Z={exc.args[0]}"
                ) from None
            cutoff = np.array([row[0] for row in rows])
            epsilon = np.array([row[1] for row in rows])
            sigma = np.array([row[2] for row in rows])
        else:
            sigma = _per_element(self.sigma, elements)  # type: ignore[arg-type]
            epsilon = _per_element(self.epsilon, elements)  # type: ignore[arg-type]
            cutoff = None

        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be positive")
        if np.any(epsilon < 0.0):
            raise ValueError("epsilon must be non-negative")

        values = {"r0": SIGMA_TO_R0 * sigma, "depth": epsilon}
        return PairParameters(
            elements=np.asarray(elements, dtype=np.int64),
            values={
                name: to_backend(value, self.backend)
                for name, value in values.items()
            },
            mixing=self.mixing,
            cutoff=None if cutoff is None else to_backend(cutoff, self.backend),
        )
