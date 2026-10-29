"""Generic trainable wrapper for pairwise potentials."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from arrayff.potentials.pairwise import LJ93
from arrayff.potentials.pairwise.morse import Morse
from arrayff.potentials.pairwise.params import PairParameters

__all__ = ["PairwiseTorch"]


class PairwiseTorch(nn.Module):
    """Trainable pairwise potential.

    Composes a :class:`PairwisePotential` (built on the torch backend) with
    log-space ``nn.Parameter`` per element parameters, so any form works
    without duplicating the energy expression.
    """

    potential_cls: type
    potential_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        species: list[int] | tuple[int, ...],
        initial: dict[int, dict[str, float]],
        *,
        cutoff: float = 10.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        nn.Module.__init__(self)
        self.elements = sorted(int(z) for z in species)
        self.potential = self.potential_cls(
            **self.potential_kwargs,
            cutoff=cutoff,
            backend="torch",
            dtype=dtype,
        )
        names = self.potential.form.param_names
        missing = [
            (z, name)
            for z in self.elements
            for name in names
            if name not in initial.get(z, {})
        ]
        if missing:
            raise KeyError(f"Missing initial parameters: {missing}")
        self.raw = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.log(
                        torch.as_tensor(
                            [initial[z][name] for z in self.elements],
                            dtype=dtype,
                        )
                    )
                )
                for name in names
            }
        )

    def parameters_for_pairs(self) -> PairParameters:
        return PairParameters(
            elements=np.asarray(self.elements, dtype=np.int64),
            values={name: torch.exp(value) for name, value in self.raw.items()},
            mixing=self.potential.mixing,
            cutoff=None,
        )

    def energy_tensor(self, atoms: Any) -> torch.Tensor:
        positions, species, table, cell, pbc, _ = self.potential._context(atoms)
        return self.potential.energy_arrays(
            positions, species, table, cell, pbc, self.parameters_for_pairs()
        )

    def forces_tensor(self, atoms: Any) -> torch.Tensor:
        positions, species, table, cell, pbc, _ = self.potential._context(atoms)
        positions = positions.detach().clone().requires_grad_(True)
        energy = self.potential.energy_arrays(
            positions, species, table, cell, pbc, self.parameters_for_pairs()
        )
        (gradient,) = torch.autograd.grad(
            energy, positions, create_graph=self.training
        )
        return -gradient

    def forward(self, atoms: Any) -> torch.Tensor:
        return self.energy_tensor(atoms)


class MorseTorch(PairwiseTorch):
    """Trainable Morse potential."""

    potential_cls = Morse  # type: ignore[override]
    potential_kwargs = {"r0": 1.0, "depth": 1.0, "a": 1.0}


class LJ93Torch(PairwiseTorch):
    """Trainable Lennard-Jones 9-3 potential."""

    potential_cls = LJ93  # type: ignore[override]
    potential_kwargs = {"r0": 1.0, "depth": 1.0}
