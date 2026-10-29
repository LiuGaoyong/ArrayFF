"""Lennard-Jones 9-3 surface potential."""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.potentials.pairwise.base import PairwisePotential
from arrayff.potentials.pairwise.forms import GeneralizedLJ
from arrayff.potentials.pairwise.params import PairParameters
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import to_backend

__all__ = ["LennardJones93"]


def _per_element(
    value: float | dict[int, float], elements: np.ndarray
) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([float(value[int(z)]) for z in elements])
    return np.full(len(elements), float(value))


@register_potential("lj93")
class LennardJones93(PairwisePotential):
    """9-3 Lennard-Jones potential, usually used for gas-surface pairs."""

    name = "lj93"
    form = GeneralizedLJ(9, 3)

    def __init__(
        self,
        r0: float | dict[int, float],
        depth: float | dict[int, float],
        cutoff: float = 10.0,
        *,
        shift: bool = False,
        smooth: bool = False,
        layout: str = "edges",
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
        builder: str = "sklearn",
        skin: float = 0.0,
    ) -> None:
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
        self.r0 = r0
        self.depth = depth
        self.mixing = {"r0": "arithmetic", "depth": "geometric"}

    def element_parameters(self, elements: np.ndarray) -> PairParameters:
        return PairParameters(
            elements=np.asarray(elements, dtype=np.int64),
            values={
                "r0": to_backend(_per_element(self.r0, elements), self.backend),
                "depth": to_backend(
                    _per_element(self.depth, elements), self.backend
                ),
            },
            mixing=self.mixing,
            cutoff=None,
        )
