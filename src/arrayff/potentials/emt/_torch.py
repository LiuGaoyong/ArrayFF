"""Optional torch wrapper for the effective medium theory potential."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from arrayff.potentials._torch import TorchPotential
from arrayff.potentials.emt.emt import EMT

__all__ = ["EMTTorch"]


class EMTTorch(EMT, TorchPotential):
    """Torch module version of :class:`arrayff.potentials.EMT`.

    The EMT parameters are tabulated per element and kept frozen; the module
    exists so EMT energies and forces can participate in torch graphs (e.g.
    as a reference model in a training loss).
    """

    def __init__(
        self,
        *,
        max_neighbors: int | None = None,
        dtype: torch.dtype = torch.float64,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        EMT.__init__(
            self,
            backend="torch",
            dtype=dtype,
            max_neighbors=max_neighbors,
            **kwargs,
        )
