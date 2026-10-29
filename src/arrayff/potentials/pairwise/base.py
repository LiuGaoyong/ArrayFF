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

from arrayff.neighbors._table import NeighborEdgeList, NeighborPaddedTable
from arrayff.potentials.base import Potential
from arrayff.potentials.pairwise._pairs import pair_distances, pair_vectors
from arrayff.potentials.pairwise.forms import MIXERS, PairForm
from arrayff.potentials.pairwise.params import PairParameters
from arrayff.utils.dispatch import do

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


def _switch_derivative(r: Any, cutoff: Any) -> Any:
    """Derivative of :func:`_switch` with respect to ``r``."""
    inner = 0.9 * cutoff
    width = cutoff - inner
    x = (r - inner) / width
    x = do("where", r <= inner, 0.0, x)
    x = do("where", r >= cutoff, 1.0, x)
    derivative = -30.0 * x * x * (1.0 - x) * (1.0 - x) / width
    derivative = do("where", r <= inner, 0.0, derivative)
    derivative = do("where", r >= cutoff, 0.0, derivative)
    return derivative


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
        self._analytic_padded: NeighborPaddedTable | None = None
        self._analytic_source: NeighborEdgeList | None = None

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

    # -- energy (AD path) ----------------------------------------------------
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
        cutoff = self._pair_cutoff(first, second, params)

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

    def _pair_cutoff(
        self, first: Any, second: Any, params: PairParameters
    ) -> Any:
        if params.cutoff is None:
            return self.cutoff
        rule = params.mixing.get("cutoff", "geometric")
        return MIXERS[rule](params.cutoff[first], params.cutoff[second])

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
            r = pair_distances(positions, table, cell)
            first = species[:, None]
            second = species[table.index]
            mask = table.mask
        return self._pair_terms(r, first, second, mask, params)

    # -- analytic path -------------------------------------------------------
    def _flatten_params(
        self, params: PairParameters
    ) -> tuple[tuple[Any, ...], Any]:
        names = self.form.param_names
        values = tuple(params.values[name] for name in names)
        has_cutoff = params.cutoff is not None
        if has_cutoff:
            values = values + (params.cutoff,)

        def rebuild(vals: tuple[Any, ...]) -> PairParameters:
            names_vals = dict(zip(names, vals[: len(names)]))
            cutoff = vals[-1] if has_cutoff else None
            return PairParameters(
                elements=params.elements,
                values=names_vals,
                mixing=params.mixing,
                cutoff=cutoff,
            )

        return values, rebuild

    def _analytic_table(
        self,
        table: NeighborPaddedTable | NeighborEdgeList,
        n_atoms: int,
    ) -> NeighborPaddedTable:
        if isinstance(table, NeighborPaddedTable):
            return table
        if self._analytic_source is not table:
            self._analytic_padded = table.to_padded(
                n_atoms, max_neighbors=self.max_neighbors
            )
            self._analytic_source = table
        assert self._analytic_padded is not None
        return self._analytic_padded

    def _pair_derivative_terms(
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable,
        cell: Any,
        params: PairParameters,
        *,
        want_forces: bool,
        want_stress: bool,
    ) -> tuple[Any, Any, Any, Any]:
        form = self.form
        r = pair_distances(positions, table, cell)
        vectors = pair_vectors(positions, table, cell)
        first = species[:, None]
        second = species[table.index]
        mixed = form.mix(
            {name: params.values[name][first] for name in form.param_names},
            {name: params.values[name][second] for name in form.param_names},
            params.mixing,
        )
        cutoff = self._pair_cutoff(first, second, params)
        inside = do(
            "logical_and",
            table.mask,
            do("logical_and", r < cutoff, r > 0.0),
        )
        r_safe = do("where", inside, r, cutoff)

        value = form.energy(r_safe, mixed)
        derivative = form.d_energy_dr(r_safe, mixed)
        cutoff_energy = form.cutoff_energy(cutoff, mixed)
        if self.shift:
            value = value - cutoff_energy
        if self.smooth:
            switch = _switch(r_safe, cutoff)
            switch_derivative = _switch_derivative(r_safe, cutoff)
            # ``value`` already contains the shift subtraction (if any).
            derivative = derivative * switch + value * switch_derivative
            value = value * switch
        value = do("where", inside, value, 0.0)
        derivative = do("where", inside, derivative, 0.0)
        energy = 0.5 * do("sum", value)
        coefficient = do("where", inside, derivative / r_safe, 0.0)

        forces = None
        if want_forces:
            # F_i = -dV/dr * d r / d r_i = +dV/dr * (r_j - r_i) / r
            forces = do(
                "sum", coefficient[:, :, None] * vectors, axis=1
            )

        stress = None
        if want_stress:
            virial = 0.5 * do(
                "einsum", "nm,nmi,nmj->ij", coefficient, vectors, vectors
            )
            volume = do("abs", do("linalg.det", cell))
            stress = virial / volume
        return energy, forces, stress, inside

    def evaluate_arrays(  # type: ignore[override]
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: PairParameters,
        *,
        want: tuple[str, ...] = ("energy", "forces", "stress"),
    ) -> dict[str, Any]:
        table = self._analytic_table(table, int(positions.shape[0]))
        energy, forces, stress, _ = self._pair_derivative_terms(
            positions,
            species,
            table,
            cell,
            params,
            want_forces="forces" in want,
            want_stress="stress" in want,
        )
        out: dict[str, Any] = {}
        if "energy" in want:
            out["energy"] = energy
        if forces is not None:
            out["forces"] = forces
        if stress is not None:
            out["stress"] = stress
        return out

    def forces_arrays(  # type: ignore[override]
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: PairParameters,
    ) -> Any:
        return self.evaluate_arrays(
            positions, species, table, cell, pbc, params,
            want=("forces",),
        )["forces"]

    def stress_arrays(  # type: ignore[override]
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: PairParameters,
    ) -> Any:
        return self.evaluate_arrays(
            positions, species, table, cell, pbc, params,
            want=("stress",),
        )["stress"]

    def clear_cache(self) -> None:
        super().clear_cache()
        self._analytic_padded = None
        self._analytic_source = None
