"""Generic pairwise potential.

Only the functional form and the per element parameters are specific to a
pairwise potential; everything else (neighbour layout, cutoffs, shifting,
smoothing, forces, virial stress) lives here.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
from ase import Atoms

from arrayff import ad
from arrayff.neighbors._table import NeighborEdgeList, NeighborPaddedTable
from arrayff.potentials.base import Potential
from arrayff.potentials.pairwise.forms import MIXERS, PairForm
from arrayff.potentials.pairwise.params import PairParameters
from arrayff.utils.dispatch import do, to_numpy

__all__ = ["PairwisePotential"]


def _switch(r: Any, cutoff: Any) -> Any:
    """Smooth C1 switch that goes to zero at ``cutoff``.

    Active over the last 10% of the cutoff; identity elsewhere.
    """
    inner = 0.9 * cutoff
    width = cutoff - inner
    x = (r - inner) / width
    x = do("where", r <= inner, 0.0, x)
    x = do("where", r >= cutoff, 1.0, x)
    return 1.0 - (x * x * x * (10.0 + x * (-15.0 + 6.0 * x)))


class PairwisePotential(Potential):
    """Base class for pair potentials.

    Parameters
    ----------
    cutoff:
        Global cutoff radius (Angstrom). Ignored per pair when the parameter
        set carries per element cutoffs (OpenKIM universal model).
    shift:
        Shift each pair energy to zero at its cutoff.
    smooth:
        Multiply by a C1 switch over the last 10% of the cutoff.
    layout:
        ``"edges"`` (default) uses :class:`NeighborEdgeList`; ``"padded"``
        uses :class:`NeighborPaddedTable` for static shapes (JAX jit).
    """

    form: PairForm

    def __init__(
        self,
        cutoff: float,
        *,
        shift: bool = True,
        smooth: bool = False,
        layout: str = "edges",
        **kwargs: Any,
    ) -> None:
        super().__init__(cutoff, **kwargs)
        if layout not in ("edges", "padded"):
            raise ValueError("layout must be 'edges' or 'padded'")
        self.shift = bool(shift)
        self.smooth = bool(smooth)
        self.layout = layout

    # -- hooks ---------------------------------------------------------------
    @abc.abstractmethod
    def element_parameters(  # type: ignore[override]
        self,
        elements: np.ndarray,
    ) -> PairParameters:
        """Return the per element parameters of this potential."""

    def _build_table(  # type: ignore[override]
        self,
        atoms: Atoms,
        cutoff: float,
    ) -> NeighborPaddedTable | NeighborEdgeList:
        if self.layout == "padded":
            return NeighborPaddedTable.from_ase(
                atoms,
                cutoff,
                max_neighbors=self.max_neighbors,
                backend=self.backend,
                builder=self.builder,
            )
        return NeighborEdgeList.from_ase(
            atoms, cutoff, backend=self.backend, builder=self.builder
        )

    # -- distance helpers ----------------------------------------------------
    def _edge_distances(
        self, positions: Any, table: NeighborEdgeList, cell: Any
    ):
        source = table.source
        target = table.target
        vectors = positions[target] - positions[source]
        if cell is not None:
            from arrayff.utils.dispatch import dtype_of

            shift = do("astype", table.shift, dtype_of(positions))
            vectors = vectors + do("einsum", "ek,kl->el", shift, cell)
        r = do("sqrt", do("sum", vectors * vectors, axis=-1))
        return r, vectors

    # -- energy --------------------------------------------------------------
    def _pair_terms(
        self,
        r: Any,
        first: Any,
        second: Any,
        mask: Any,
        params: PairParameters,
    ) -> Any:
        form = self.form
        mixed = form.mix(
            {name: params.values[name][first] for name in form.param_names},
            {name: params.values[name][second] for name in form.param_names},
            params.mixing,
        )
        if params.cutoff is not None:
            rule = params.mixing.get("cutoff", "geometric")
            cutoff = MIXERS[rule](params.cutoff[first], params.cutoff[second])
        else:
            cutoff = self.cutoff

        inside = r < cutoff
        if mask is not None:
            inside = do("logical_and", mask, inside)
        r_safe = do("where", inside, r, cutoff)
        energy = form.energy(r_safe, mixed)
        if self.shift:
            energy = energy - form.cutoff_energy(cutoff, mixed)
        if self.smooth:
            energy = energy * _switch(r_safe, cutoff)
        energy = do("where", inside, energy, 0.0)
        return 0.5 * do("sum", energy)

    def energy_arrays(  # type: ignore[override]
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: PairParameters,
    ) -> Any:
        if isinstance(table, NeighborEdgeList):
            r, _ = self._edge_distances(positions, table, cell)
            first = species[table.source]
            second = species[table.target]
            mask = None
        else:
            from arrayff.potentials.pairwise._pairs import pair_distances

            r = pair_distances(positions, table, cell)
            first = species[:, None]
            second = species[table.index]
            mask = table.mask
        return self._pair_terms(r, first, second, mask, params)

    # -- stress --------------------------------------------------------------
    def stress(self, atoms: Atoms, step: float = 1e-5) -> np.ndarray:
        """Virial stress from the pair forces (exact for pair potentials)."""
        if atoms.cell.rank != 3:
            raise ValueError("stress requires a fully periodic cell")
        if self.layout != "edges":
            return super().stress(atoms, step=step)
        positions, species, table, cell, pbc, params = self._context(atoms)
        if len(table) == 0:
            return np.zeros((3, 3))
        r, vectors = self._edge_distances(positions, table, cell)

        form = self.form
        first = species[table.source]
        second = species[table.target]
        mixed = form.mix(
            {name: params.values[name][first] for name in form.param_names},
            {name: params.values[name][second] for name in form.param_names},
            params.mixing,
        )
        if params.cutoff is not None:
            rule = params.mixing.get("cutoff", "geometric")
            cutoff = MIXERS[rule](params.cutoff[first], params.cutoff[second])
        else:
            cutoff = self.cutoff
        inside = r < cutoff

        def total(rr: Any) -> Any:
            return do("sum", form.energy(rr, mixed))

        derivative = ad.grad(total)(r)
        coefficient = do("where", inside, derivative / r, 0.0)
        virial = do("einsum", "e,ei,ej->ij", coefficient, vectors, vectors)
        volume = float(atoms.get_volume())
        return 0.5 * to_numpy(virial) / volume
