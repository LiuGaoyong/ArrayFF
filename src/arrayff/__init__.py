"""ArrayFF: force fields on multiple array backends.

The package is organised as a thin dispatch layer (:mod:`arrayff.utils`),
backend agnostic autodiff (:mod:`arrayff.utils.ad`) and numerical kernels that
never import a concrete array library.
"""

from __future__ import annotations

from arrayff import utils
from arrayff.electrostatics import QEq
from arrayff.potentials import (
    CompositePotential,
    LennardJones,
    available_potentials,
    get_potential,
)

__version__ = "0.0.1"

__all__ = [
    "CompositePotential",
    "LennardJones",
    "QEq",
    "__version__",
    "available_potentials",
    "get_potential",
    "utils",
]
