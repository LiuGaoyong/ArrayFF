"""Short ranged potentials and their optional adapters.

The core potential classes are imported eagerly.  ASE calculators and torch
wrappers are exposed lazily: importing ``arrayff.potentials`` must not import
``torch`` or ``ase.calculators`` on their behalf.
"""

from __future__ import annotations

import importlib
from typing import Any

from arrayff.potentials.base import Potential, PotentialBase
from arrayff.potentials.composite import CompositePotential
from arrayff.potentials.eam.eam import EAM
from arrayff.potentials.emt.emt import EMT
from arrayff.potentials.pairwise.lj import LennardJones
from arrayff.potentials.pairwise.lj93 import LennardJones93 as LJ93
from arrayff.potentials.pairwise.morse import Morse
from arrayff.potentials.placeholders import MEAM, UF3
from arrayff.potentials.registry import (
    available_potentials,
    get_potential,
    register_potential,
)

__all__ = [
    "EAM",
    "EMT",
    "LJ93",
    "MEAM",
    "UF3",
    "ArrayFFCalculator",
    "CompositePotential",
    "EAMCalculator",
    "EMTCalculator",
    "LJ93Calculator",
    "LJ93Torch",
    "LJCalculator",
    "LJTorch",
    "LennardJones",
    "Morse",
    "MorseCalculator",
    "MorseTorch",
    "PairwiseTorch",
    "Potential",
    "PotentialBase",
    "available_potentials",
    "get_potential",
    "register_potential",
]

_LAZY: dict[str, str] = {
    "ArrayFFCalculator": "arrayff.potentials.calculator",
    "LJCalculator": "arrayff.potentials.pairwise.calculator",
    "MorseCalculator": "arrayff.potentials.pairwise.calculator",
    "LJ93Calculator": "arrayff.potentials.pairwise.calculator",
    "EMTCalculator": "arrayff.potentials.emt.calculator",
    "EAMCalculator": "arrayff.potentials.eam.calculator",
    "LJTorch": "arrayff.potentials.pairwise._torch",
    "MorseTorch": "arrayff.potentials.pairwise._torch",
    "LJ93Torch": "arrayff.potentials.pairwise._torch",
    "PairwiseTorch": "arrayff.potentials.pairwise._torch",
    "EMTTorch": "arrayff.potentials.emt._torch",
    "EAMTorch": "arrayff.potentials.eam._torch",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
