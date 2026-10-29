"""Shared tensor level API for the optional torch wrappers.

Importing this module is the opt-in point for PyTorch: ``arrayff.potentials``
itself never imports it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["TorchPotential"]


class TorchPotential(nn.Module):
    """Mixin adding tensor energy/forces on top of a :class:`Potential`."""

    def energy_tensor(self, atoms: Any) -> torch.Tensor:
        """Return the energy as a scalar tensor (graph attached)."""
        positions, species, table, cell, pbc, params = self._context(atoms)
        return self.energy_arrays(positions, species, table, cell, pbc, params)

    def forces_tensor(
        self, atoms: Any, *, method: str = "autodiff"
    ) -> torch.Tensor:
        """Return forces (eV/Angstrom) as a tensor."""
        positions, species, table, cell, pbc, params = self._context(atoms)
        if method == "analytic":
            return self.forces_arrays(
                positions, species, table, cell, pbc, params
            )
        positions = positions.detach().clone().requires_grad_(True)
        energy = self.energy_arrays(
            positions, species, table, cell, pbc, params
        )
        (gradient,) = torch.autograd.grad(
            energy, positions, create_graph=self.training
        )
        return -gradient

    def forward(self, atoms: Any) -> torch.Tensor:
        return self.energy_tensor(atoms)
