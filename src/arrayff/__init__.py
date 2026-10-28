"""ArrayFF: force fields on multiple array backends.

The package is organised as a thin dispatch layer (:mod:`arrayff.utils`),
backend agnostic autodiff (:mod:`arrayff.ad`) and numerical kernels that never
import a concrete array library.
"""

from __future__ import annotations

from arrayff import ad, utils
from arrayff.potentials import (
    LennardJones,
    available_potentials,
    get_potential,
)

__version__ = "0.0.1"

__all__ = [
    "LennardJones",
    "__version__",
    "ad",
    "available_potentials",
    "get_potential",
    "utils",
]
