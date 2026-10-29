"""ASE calculator for tabulated EAM potentials."""

from __future__ import annotations

from typing import Any

from arrayff.potentials.calculator import ArrayFFCalculator, split_kwargs
from arrayff.potentials.eam.eam import EAM

__all__ = ["EAMCalculator"]


class EAMCalculator(ArrayFFCalculator):
    """EAM calculator reading a ``setfl`` file."""

    def __init__(self, potential: str, **kwargs: Any) -> None:
        potential_kwargs, calculator_kwargs = split_kwargs(kwargs, EAM)
        super().__init__(
            EAM(potential, **potential_kwargs), **calculator_kwargs
        )
