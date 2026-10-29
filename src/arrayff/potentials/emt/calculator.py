"""ASE calculator for the effective medium theory potential."""

from __future__ import annotations

from typing import Any

from arrayff.potentials.calculator import ArrayFFCalculator, split_kwargs
from arrayff.potentials.emt.emt import EMT

__all__ = ["EMTCalculator"]


class EMTCalculator(ArrayFFCalculator):
    """Effective medium theory calculator."""

    def __init__(self, **kwargs: Any) -> None:
        potential_kwargs, calculator_kwargs = split_kwargs(kwargs, EMT)
        super().__init__(EMT(**potential_kwargs), **calculator_kwargs)
