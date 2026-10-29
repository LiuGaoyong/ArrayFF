"""Trainable Lennard-Jones module."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from arrayff.nn.base import TorchPotential
from arrayff.potentials.pairwise.lj import KIM_LJ_PARAMETERS, LennardJones

__all__ = ["LJTorch"]


class LJTorch(LennardJones, TorchPotential):
    """Lennard-Jones with trainable per element ``epsilon``/``sigma``.

    Parameters are stored in log space so an optimizer can never drive them
    negative; the initial values reproduce the OpenKIM universal model.
    """

    def __init__(
        self,
        species: list[int] | tuple[int, ...],
        *,
        max_neighbors: int | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        nn.Module.__init__(self)
        LennardJones.__init__(
            self,
            backend="torch",
            dtype=dtype,
            max_neighbors=max_neighbors,
        )
        self.elements = sorted(int(z) for z in species)
        epsilon = np.array([KIM_LJ_PARAMETERS[z][1] for z in self.elements])
        sigma = np.array([KIM_LJ_PARAMETERS[z][2] for z in self.elements])
        cutoff = np.array([KIM_LJ_PARAMETERS[z][0] for z in self.elements])
        self.raw_epsilon = nn.Parameter(
            torch.log(torch.as_tensor(epsilon, dtype=dtype))
        )
        self.raw_sigma = nn.Parameter(
            torch.log(torch.as_tensor(sigma, dtype=dtype))
        )
        self.register_buffer(
            "cutoff_table", torch.as_tensor(cutoff, dtype=dtype)
        )

    def element_parameters(self, elements: np.ndarray) -> dict[str, Any]:
        found = [int(z) for z in elements]
        if found != self.elements:
            raise ValueError(
                f"LJTorch was built for elements {self.elements}, got {found}"
            )
        return {
            "epsilon": torch.exp(self.raw_epsilon),
            "sigma": torch.exp(self.raw_sigma),
            "cutoff": self.cutoff_table,
        }

    def forward(self, atoms: Any) -> torch.Tensor:
        return self.energy_tensor(atoms)
