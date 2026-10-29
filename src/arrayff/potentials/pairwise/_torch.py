"""Optional trainable torch wrappers for pairwise potentials."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from arrayff.potentials._torch import TorchPotential
from arrayff.potentials.pairwise.lj import (
    KIM_LJ_PARAMETERS,
    SIGMA_TO_R0,
    LennardJones,
)
from arrayff.potentials.pairwise.lj93 import LennardJones93
from arrayff.potentials.pairwise.morse import Morse
from arrayff.potentials.pairwise.params import PairParameters

__all__ = [
    "LJ93Torch",
    "LJTorch",
    "MorseTorch",
    "PairwiseTorch",
]


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

    def forces_tensor(
        self, atoms: Any, *, method: str = "autodiff"
    ) -> torch.Tensor:
        positions, species, table, cell, pbc, _ = self.potential._context(atoms)
        if method == "analytic":
            return self.potential.forces_arrays(
                positions, species, table, cell, pbc,
                self.parameters_for_pairs(),
            )
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

    potential_cls = LennardJones93  # type: ignore[override]
    potential_kwargs = {"r0": 1.0, "depth": 1.0}


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

    def element_parameters(self, elements: np.ndarray) -> PairParameters:
        found = [int(z) for z in elements]
        if found != self.elements:
            raise ValueError(
                f"LJTorch was built for elements {self.elements}, got {found}"
            )
        return PairParameters(
            elements=np.asarray(elements, dtype=np.int64),
            values={
                "r0": SIGMA_TO_R0 * torch.exp(self.raw_sigma),
                "depth": torch.exp(self.raw_epsilon),
            },
            mixing=self.mixing,
            cutoff=self.cutoff_table,
        )

    def forward(self, atoms: Any) -> torch.Tensor:
        return self.energy_tensor(atoms)
