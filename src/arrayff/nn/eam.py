"""Torch wrapper for tabulated EAM potentials."""

from __future__ import annotations


import torch
from torch import nn

from arrayff.nn.base import TorchPotential
from arrayff.potentials.eam import EAM

__all__ = ["EAMTorch"]


class EAMTorch(EAM, TorchPotential):
    """Torch module version of :class:`arrayff.potentials.EAM`."""

    def __init__(
        self,
        potential: str,
        *,
        max_neighbors: int | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        nn.Module.__init__(self)
        EAM.__init__(
            self,
            potential,
            backend="torch",
            dtype=dtype,
            max_neighbors=max_neighbors,
        )
