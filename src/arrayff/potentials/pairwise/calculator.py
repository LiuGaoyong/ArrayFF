"""ASE calculators for the pairwise potentials."""

from __future__ import annotations

from typing import Any

from arrayff.potentials.calculator import ArrayFFCalculator, split_kwargs
from arrayff.potentials.pairwise.lj import LennardJones
from arrayff.potentials.pairwise.lj93 import LennardJones93
from arrayff.potentials.pairwise.morse import Morse

__all__ = ["LJ93Calculator", "LJCalculator", "MorseCalculator"]


class LJCalculator(ArrayFFCalculator):
    """Lennard-Jones calculator (OpenKIM universal parameters by default)."""

    def __init__(self, **kwargs: Any) -> None:
        potential_kwargs, calculator_kwargs = split_kwargs(kwargs, LennardJones)  # type: ignore
        super().__init__(LennardJones(**potential_kwargs), **calculator_kwargs)
        self.parameters = potential_kwargs


class MorseCalculator(ArrayFFCalculator):
    """Morse calculator."""

    def __init__(self, **kwargs: Any) -> None:
        potential_kwargs, calculator_kwargs = split_kwargs(kwargs, Morse)  # type: ignore
        super().__init__(Morse(**potential_kwargs), **calculator_kwargs)


class LJ93Calculator(ArrayFFCalculator):
    """Lennard-Jones 9-3 calculator."""

    def __init__(self, **kwargs: Any) -> None:
        potential_kwargs, calculator_kwargs = split_kwargs(
            kwargs,
            LennardJones93,  # type: ignore
        )
        super().__init__(
            LennardJones93(**potential_kwargs),
            **calculator_kwargs,
        )
