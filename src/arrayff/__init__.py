"""ArrayFF: force fields on multiple array backends.

The package is organised as a thin dispatch layer (:mod:`arrayff.utils`),
backend agnostic autodiff (:mod:`arrayff.ad`) and numerical kernels that never
import a concrete array library.
"""

from __future__ import annotations

from arrayff import ad, utils

__version__ = "0.0.1"

__all__ = ["__version__", "ad", "utils"]
