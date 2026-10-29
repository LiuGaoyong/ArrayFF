"""ASE calculator for the QEq charge model."""

from __future__ import annotations

from typing import Any

from arrayff.electrostatics.qeq import QEq
from arrayff.potentials.calculator import ArrayFFCalculator, split_kwargs

__all__ = ["QEqCalculator"]


class QEqCalculator(ArrayFFCalculator):
    """Charge equilibration calculator (exposes ``charges``)."""

    def __init__(self, **kwargs: Any) -> None:
        potential_kwargs, calculator_kwargs = split_kwargs(kwargs, QEq)
        super().__init__(QEq(**potential_kwargs), **calculator_kwargs)
