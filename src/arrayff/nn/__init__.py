"""Torch wrappers for arrayff potentials (optional dependency).

Importing :mod:`arrayff.nn` requires torch; the rest of arrayff does not.
The wrappers reuse the same autoray kernels as the plain potentials, so there
is no duplicated physics: only the parameter storage differs (trainable
``nn.Parameter`` tensors instead of dictionaries).
"""

from __future__ import annotations

from arrayff.utils.exceptions import MissingBackendError

try:  # pragma: no cover - exercised by the torch test
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise MissingBackendError(
        "arrayff.nn requires PyTorch; "
        "install it with `pip install arrayff[torch]`"
    ) from exc

from arrayff.nn.base import TorchPotential
from arrayff.nn.eam import EAMTorch
from arrayff.nn.emt import EMTTorch
from arrayff.nn.lj import LJTorch

__all__ = ["EAMTorch", "EMTTorch", "LJTorch", "TorchPotential", "nn", "torch"]
