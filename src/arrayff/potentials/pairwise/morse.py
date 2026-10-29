"""Morse potential built on the generic pairwise layer."""

from __future__ import annotations

from typing import Any

import numpy as np

from arrayff.potentials.pairwise.base import PairwisePotential
from arrayff.potentials.pairwise.forms import Morse as MorseForm
from arrayff.potentials.pairwise.params import PairParameters
from arrayff.potentials.registry import register_potential
from arrayff.utils.dispatch import to_backend

__all__ = ["Morse"]


def _per_element(
    value: float | dict[int, float], elements: np.ndarray
) -> np.ndarray:
    if isinstance(value, dict):
        missing = [int(z) for z in elements if int(z) not in value]
        if missing:
            raise KeyError(f"Missing Morse parameters for Z={missing}")
        return np.array([float(value[int(z)]) for z in elements])
    return np.full(len(elements), float(value))


@register_potential("morse")
class Morse(PairwisePotential):
    """Morse potential ``depth * (exp(-2a(r-r0)) - 2 exp(-a(r-r0)))``."""

    name = "morse"
    form = MorseForm()

    def __init__(
        self,
        r0: float | dict[int, float],
        depth: float | dict[int, float],
        a: float | dict[int, float] = 1.5,
        cutoff: float = 10.0,
        *,
        shift: bool = True,
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
        self.a = a
        self.mixing = {
            "r0": "arithmetic",
            "depth": "geometric",
            "a": "arithmetic",
        }

    def element_parameters(self, elements: np.ndarray) -> PairParameters:
        values = {
            "r0": _per_element(self.r0, elements),
            "depth": _per_element(self.depth, elements),
            "a": _per_element(self.a, elements),
        }
        if np.any(values["r0"] <= 0.0):
            raise ValueError("r0 must be positive")
        if np.any(values["depth"] < 0.0):
            raise ValueError("depth must be non-negative")
        if np.any(values["a"] <= 0.0):
            raise ValueError("a must be positive")
        return PairParameters(
            elements=np.asarray(elements, dtype=np.int64),
            values={
                name: to_backend(value, self.backend)
                for name, value in values.items()
            },
            mixing={
                "r0": "arithmetic",
                "depth": "geometric",
                "a": "arithmetic",
            },
            cutoff=None,
        )
