"""Base classes shared by all potentials.

A potential is defined by array level functions written with
:func:`arrayff.utils.dispatch.do`.  :class:`PotentialBase` is the minimal
protocol shared by table-driven potentials, QEq and composite potentials;
:class:`Potential` adds the neighbour-table machinery and the analytic/AD
evaluation pipeline.
"""

from __future__ import annotations

import abc
import hashlib
import warnings
from typing import Any

import numpy as np
from ase import Atoms

from arrayff.neighbors._table import NeighborEdgeList, NeighborPaddedTable
from arrayff.utils import ad
from arrayff.utils.ase import atoms_to_arrays, species_of
from arrayff.utils.dispatch import do, to_backend, to_numpy
from arrayff.utils.exceptions import ScopeError
from arrayff.utils.jit import autojit

__all__ = ["Potential", "PotentialBase"]

_METHODS = ("analytic", "autodiff", "finite-difference")


class PotentialBase(abc.ABC):
    """Minimal protocol for anything that can be added to a potential."""

    backend: str
    dtype: Any
    supports_charges: bool = False

    def __init__(
        self,
        *,
        backend: str = "numpy",
        dtype: Any = None,
        jit: bool = False,
        method: str = "analytic",
    ) -> None:
        self.backend = backend
        self.dtype = dtype
        self.jit = bool(jit)
        if method not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS}, got {method!r}"
            )
        self.method = method

    # -- evaluation ----------------------------------------------------------
    @abc.abstractmethod
    def evaluate(
        self,
        atoms: Atoms,
        *,
        properties: tuple[str, ...] = ("energy",),
        method: str | None = None,
    ) -> dict[str, Any]:
        """Return the requested properties as a dictionary."""

    def _resolve_method(self, method: str | None) -> str:
        resolved = self.method if method is None else method
        if resolved not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS}, got {resolved!r}"
            )
        return resolved

    def energy(self, atoms: Atoms, *, method: str | None = None) -> float:
        result = self.evaluate(atoms, properties=("energy",), method=method)
        return float(result["energy"])

    def energy_and_forces(
        self, atoms: Atoms, *, method: str | None = None
    ) -> tuple[float, np.ndarray]:
        result = self.evaluate(
            atoms, properties=("energy", "forces"), method=method
        )
        return float(result["energy"]), np.asarray(result["forces"])

    def stress(
        self, atoms: Atoms, *, method: str | None = None
    ) -> np.ndarray:
        result = self.evaluate(atoms, properties=("stress",), method=method)
        return np.asarray(result["stress"])

    def charges(self, atoms: Atoms) -> np.ndarray:
        raise ScopeError(
            f"{type(self).__name__} does not provide atomic charges"
        )

    def clear_cache(self) -> None:
        """Drop cached neighbour tables / compiled kernels."""

    # -- composition ---------------------------------------------------------
    def __add__(self, other: Any) -> Any:
        if not isinstance(other, PotentialBase):
            return NotImplemented
        from arrayff.potentials.composite import CompositePotential

        return CompositePotential((self, other))

    def __radd__(self, other: Any) -> Any:
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented


class Potential(PotentialBase):
    """Energy/force evaluator bound to one array backend.

    Parameters
    ----------
    cutoff:
        Global cutoff radius in Angstrom used to build the neighbour table.
    backend:
        ``"numpy"`` (default), ``"torch"`` or ``"jax"``.
    dtype:
        Floating point dtype used when converting the ``Atoms`` positions.
    max_neighbors:
        Fixed neighbour-table width; required to keep shapes static for JAX.
    builder:
        Neighbour list implementation: ``"sklearn"`` (default), ``"vesin"``,
        ``"matscipy"``, ``"pymatgen"`` or ``"ase_kdtree"``.
    jit:
        Compile the array level kernels with :func:`arrayff.utils.jit`.
    method:
        Default evaluation method: ``"analytic"``, ``"autodiff"`` or
        ``"finite-difference"`` (diagnostic only).
    """

    name: str = "potential"

    def __init__(
        self,
        cutoff: float,
        *,
        backend: str = "numpy",
        dtype: Any = None,
        max_neighbors: int | None = None,
        builder: str = "sklearn",
        skin: float = 0.0,
        jit: bool = False,
        method: str = "analytic",
    ) -> None:
        super().__init__(
            backend=backend, dtype=dtype, jit=jit, method=method
        )
        self.cutoff = float(cutoff)
        self.max_neighbors = max_neighbors
        self.builder = builder
        self.skin = float(skin)
        if self.skin < 0.0:
            raise ValueError("skin must be non-negative")
        self._table_static: tuple | None = None
        self._table_positions: np.ndarray | None = None
        self._table_cache: NeighborPaddedTable | NeighborEdgeList | None = None
        self._jit_cache: dict[tuple, Any] = {}

    # -- to be implemented by subclasses ---------------------------------------
    @abc.abstractmethod
    def element_parameters(self, elements: np.ndarray) -> Any:
        """Return backend arrays of the parameters for ``elements``."""

    @abc.abstractmethod
    def energy_arrays(
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: Any,
    ) -> Any:
        """Return the total energy as a scalar array."""

    def forces_arrays(
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: Any,
    ) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytic forces"
        )

    def stress_arrays(
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: Any,
    ) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytic stress"
        )

    def evaluate_arrays(
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: Any,
        *,
        want: tuple[str, ...] = ("energy", "forces", "stress"),
    ) -> dict[str, Any]:
        """Analytic array-level evaluation.

        Subclasses that can share intermediates between energy, forces and
        stress override this method; the default keeps the three kernels
        separate.
        """
        out: dict[str, Any] = {}
        if "energy" in want:
            out["energy"] = self.energy_arrays(
                positions, species, table, cell, pbc, params
            )
        if "forces" in want:
            out["forces"] = self.forces_arrays(
                positions, species, table, cell, pbc, params
            )
        if "stress" in want:
            out["stress"] = self.stress_arrays(
                positions, species, table, cell, pbc, params
            )
        return out

    # -- pipeline -----------------------------------------------------------
    def _cutoff_for(self, atoms: Atoms) -> float:
        """Return the neighbour table cutoff for ``atoms``."""
        return self.cutoff

    def _context(self, atoms: Atoms) -> tuple:
        elements, species_np = species_of(atoms.numbers)
        positions, _, cell, pbc = atoms_to_arrays(
            atoms, self.backend, self.dtype
        )
        species = to_backend(species_np, self.backend)
        table = self._table(atoms)
        return (
            positions,
            species,
            table,
            cell,
            pbc,
            self.element_parameters(elements),
        )

    def _table_static_key(self, atoms: Atoms) -> tuple:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray(atoms.cell.array, dtype=np.float64).tobytes())
        digest.update(np.asarray(atoms.pbc, dtype=bool).tobytes())
        digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
        return (
            digest.hexdigest(),
            self._cutoff_for(atoms),
            self.max_neighbors,
            self.builder,
            self.backend,
            self.skin,
        )

    def _table(self, atoms: Atoms) -> NeighborPaddedTable | NeighborEdgeList:
        """Return the neighbour table, cached according to ``skin``.

        * ``skin == 0``: reuse only when the geometry is bitwise unchanged.
        * ``skin > 0``: build with ``cutoff + skin`` and reuse while no atom
          moved more than ``skin / 2`` (the standard rebuild criterion).
        """
        static = self._table_static_key(atoms)
        positions = np.asarray(atoms.positions, dtype=np.float64)
        if (
            self._table_cache is not None
            and self._table_static == static
            and self._table_positions is not None
        ):
            displacement = float(
                np.linalg.norm(positions - self._table_positions, axis=1).max(
                    initial=0.0
                )
            )
            tolerance = self.skin / 2.0 if self.skin > 0.0 else 0.0
            if displacement <= tolerance:
                return self._table_cache

        table = self._build_table(atoms, self._cutoff_for(atoms) + self.skin)
        self._table_static = static
        self._table_positions = positions.copy()
        self._table_cache = table
        return table

    def _build_table(self, atoms: Atoms, cutoff: float) -> NeighborPaddedTable:
        """Build the neighbour table; pairwise potentials override this."""
        return NeighborPaddedTable.from_ase(
            atoms,
            cutoff,
            max_neighbors=self.max_neighbors,
            backend=self.backend,
            builder=self.builder,
        )

    def clear_cache(self) -> None:
        """Drop the cached neighbour table and compiled kernels."""
        self._table_cache = None
        self._table_positions = None
        self._table_static = None
        self._jit_cache.clear()

    # -- evaluation ----------------------------------------------------------
    def _analytic_table(
        self,
        table: NeighborPaddedTable | NeighborEdgeList,
        n_atoms: int,
    ) -> NeighborPaddedTable:
        return table  # type: ignore[return-value]

    # -- jit plumbing --------------------------------------------------------
    def _as_jit_arg(self, value: Any) -> Any:
        if isinstance(value, (np.ndarray,)) or hasattr(value, "shape"):
            return value
        return to_backend(np.asarray(value), self.backend)

    def _table_arrays(
        self, table: NeighborPaddedTable | NeighborEdgeList
    ) -> tuple[str, tuple[Any, ...]]:
        if isinstance(table, NeighborPaddedTable):
            return "padded", (table.index, table.shift, table.mask)
        return "edges", (
            table.source,
            table.target,
            table.shift_x,
            table.shift_y,
            table.shift_z,
        )

    def _rebuild_table(self, kind: str, arrays: tuple[Any, ...]):
        if kind == "padded":
            return NeighborPaddedTable(*arrays)
        return NeighborEdgeList(*arrays)

    def _flatten_params(
        self, params: Any
    ) -> tuple[tuple[Any, ...], Any]:
        if isinstance(params, dict):
            keys = tuple(sorted(params))
            values = tuple(self._as_jit_arg(params[key]) for key in keys)

            def rebuild(vals: tuple[Any, ...]) -> dict[str, Any]:
                return dict(zip(keys, vals))

            return values, rebuild
        raise TypeError(
            f"{type(self).__name__} does not implement jit parameter flattening"
        )

    def _jit_evaluate(
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable | NeighborEdgeList,
        cell: Any,
        pbc: Any,
        params: Any,
        want: tuple[str, ...],
    ) -> dict[str, Any]:
        kind, table_arrays = self._table_arrays(table)
        param_arrays, rebuild = self._flatten_params(params)
        key = (
            kind,
            want,
            tuple(tuple(np.shape(a)) for a in table_arrays),
            tuple(tuple(np.shape(a)) for a in param_arrays),
            tuple(np.asarray(a).dtype.str for a in table_arrays),
            tuple(np.asarray(a).dtype.str for a in param_arrays),
            tuple(np.shape(positions)),
        )
        compiled = self._jit_cache.get(key)
        if compiled is None:
            n_table = len(table_arrays)

            def kernel(*args: Any) -> tuple[Any, ...]:
                pos, spec, cel, pbc_ = args[:4]
                table_ = self._rebuild_table(
                    kind, args[4 : 4 + n_table]
                )
                params_ = rebuild(args[4 + n_table :])
                out = self.evaluate_arrays(
                    pos,
                    spec,
                    table_,
                    cel,
                    pbc_,
                    params_,
                    want=want,
                )
                return tuple(out[name] for name in want)

            compiled = autojit(kernel, backend=self.backend)
            self._jit_cache[key] = compiled
        args = (
            positions,
            species,
            cell,
            pbc,
            *table_arrays,
            *param_arrays,
        )
        try:
            values = compiled(*args)
        except Exception as exc:  # pragma: no cover - backend dependent
            warnings.warn(
                f"jit compilation/execution failed for {type(self).__name__} "
                f"on backend {self.backend!r}; falling back to eager: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._jit_cache[key] = None
            self.jit = False
            return self.evaluate_arrays(
                positions, species, table, cell, pbc, params, want=want
            )
        return dict(zip(want, values))

    def _jit_evaluate_ad(
        self,
        positions: Any,
        species: Any,
        table: NeighborPaddedTable | NeighborEdgeList,
        cell: Any,
        pbc: Any,
        params: Any,
        want: tuple[str, ...],
    ) -> dict[str, Any] | None:
        if self.backend not in ("jax", "torch") or "stress" in want:
            return None
        kind, table_arrays = self._table_arrays(table)
        param_arrays, rebuild = self._flatten_params(params)
        key = (
            "ad",
            self.backend,
            kind,
            want,
            tuple(tuple(np.shape(a)) for a in table_arrays),
            tuple(tuple(np.shape(a)) for a in param_arrays),
            tuple(np.shape(positions)),
        )
        compiled = self._jit_cache.get(key)
        if compiled is None:
            n_table = len(table_arrays)

            def energy_kernel(*args: Any) -> Any:
                pos, spec, cel, pbc_ = args[:4]
                table_ = self._rebuild_table(
                    kind, args[4 : 4 + n_table]
                )
                params_ = rebuild(args[4 + n_table :])
                return self.energy_arrays(
                    pos, spec, table_, cel, pbc_, params_
                )

            if self.backend == "jax":

                def kernel(*args: Any) -> tuple[Any, ...]:
                    pos = args[0]
                    energy, gradient = ad.value_and_grad(
                        lambda p: energy_kernel(p, *args[1:])
                    )(pos)
                    return energy, -gradient

                compiled = autojit(kernel, backend="jax")
            else:
                compiled = autojit(energy_kernel, backend="torch")
            self._jit_cache[key] = compiled
        try:
            if self.backend == "jax":
                values = compiled(
                    positions,
                    species,
                    cell,
                    pbc,
                    *table_arrays,
                    *param_arrays,
                )
                energy, forces = values
            else:
                import torch

                grad_positions = positions.detach().clone().requires_grad_(True)
                energy = compiled(
                    grad_positions,
                    species,
                    cell,
                    pbc,
                    *table_arrays,
                    *param_arrays,
                )
                if "forces" in want:
                    (gradient,) = torch.autograd.grad(
                        energy, grad_positions, create_graph=False
                    )
                    forces = -gradient
                else:
                    forces = None
        except Exception as exc:  # pragma: no cover - backend dependent
            warnings.warn(
                f"AD jit failed for {type(self).__name__}; falling back to "
                f"eager: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._jit_cache[key] = None
            return None
        if len(want) == 1 and want[0] == "forces":
            assert forces is not None
            return {"forces": forces}
        result = {"energy": energy}
        if forces is not None:
            result["forces"] = forces
        return result

    def evaluate(
        self,
        atoms: Atoms,
        *,
        properties: tuple[str, ...] = ("energy",),
        method: str | None = None,
    ) -> dict[str, Any]:
        props = tuple(properties)
        unknown = set(props) - {"energy", "forces", "stress"}
        if unknown:
            raise ScopeError(
                f"Unsupported properties for {type(self).__name__}: "
                f"{sorted(unknown)}"
            )
        if "stress" in props and atoms.cell.rank != 3:
            raise ValueError("stress requires a fully periodic cell")

        resolved = self._resolve_method(method)
        if resolved == "finite-difference":
            return self._evaluate_finite_difference(atoms, props)

        positions, species, table, cell, pbc, params = self._context(atoms)
        volume = float(atoms.get_volume()) if "stress" in props else 1.0

        if resolved == "analytic":
            analytic_table = self._analytic_table(table, len(atoms))
            if self.jit:
                result = self._jit_evaluate(
                    positions,
                    species,
                    analytic_table,
                    cell,
                    pbc,
                    params,
                    props,
                )
            else:
                result = self.evaluate_arrays(
                    positions,
                    species,
                    analytic_table,
                    cell,
                    pbc,
                    params,
                    want=props,
                )
        else:  # autodiff
            result = {}
            jitted = (
                self._jit_evaluate_ad(
                    positions, species, table, cell, pbc, params, props
                )
                if self.jit
                else None
            )
            if jitted is not None:
                result.update(jitted)
            elif "energy" in props or "forces" in props:

                def value(p: Any) -> Any:
                    return self.energy_arrays(
                        p, species, table, cell, pbc, params
                    )

                if "forces" in props:
                    energy, gradient = ad.value_and_grad(value)(positions)
                    result["energy"] = energy
                    result["forces"] = -gradient
                else:
                    result["energy"] = self.energy_arrays(
                        positions, species, table, cell, pbc, params
                    )
            if "stress" in props:
                result["stress"] = self._stress_autodiff(
                    positions,
                    species,
                    table,
                    cell,
                    pbc,
                    params,
                    volume,
                )

        converted: dict[str, Any] = {}
        if "energy" in result:
            converted["energy"] = float(to_numpy(result["energy"]))
        if "forces" in result:
            converted["forces"] = np.asarray(to_numpy(result["forces"]))
        if "stress" in result:
            converted["stress"] = np.asarray(to_numpy(result["stress"]))
        return converted

    def _stress_autodiff(
        self,
        positions: Any,
        species: Any,
        table: Any,
        cell: Any,
        pbc: Any,
        params: Any,
        volume: float,
    ) -> Any:
        """Differentiate the energy with respect to an affine strain."""
        basis = np.zeros((6, 3, 3), dtype=float)
        basis[0, 0, 0] = 1.0
        basis[1, 1, 1] = 1.0
        basis[2, 2, 2] = 1.0
        for index, (a, b) in enumerate(((1, 2), (0, 2), (0, 1))):
            basis[3 + index, a, b] = 1.0
            basis[3 + index, b, a] = 1.0
        basis_b = to_backend(basis, self.backend)
        zero = do("zeros", (6,), like=positions)

        def energy(strain_params: Any) -> Any:
            strain = do("einsum", "k,kij->ij", strain_params, basis_b)
            eye = do("eye", 3, like=positions)
            deform = eye + strain
            moved = do("matmul", positions, deform)
            deformed_cell = do("matmul", cell, deform)
            return self.energy_arrays(
                moved, species, table, deformed_cell, pbc, params
            )

        derivative = to_numpy(ad.jacrev(energy)(zero))
        derivative = np.asarray(derivative, dtype=float)
        factors = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        d = derivative * factors
        sigma = np.array(
            [
                [d[0], d[5], d[4]],
                [d[5], d[1], d[3]],
                [d[4], d[3], d[2]],
            ]
        )
        return sigma / volume

    def _evaluate_finite_difference(
        self, atoms: Atoms, properties: tuple[str, ...]
    ) -> dict[str, Any]:
        """Diagnostic finite-difference stress, analytic other properties."""
        out: dict[str, Any] = {}
        other = tuple(p for p in properties if p != "stress")
        if other:
            out.update(
                self.evaluate(atoms, properties=other, method="analytic")
            )
        if "stress" in properties:
            out["stress"] = self._stress_finite_difference(atoms)
        return out

    def _stress_finite_difference(
        self, atoms: Atoms, step: float = 1e-5
    ) -> np.ndarray:
        volume = float(atoms.get_volume())
        positions = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        identity = np.eye(3)
        voigt = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        derivative = np.zeros(6)
        try:
            self.clear_cache()
            for index, (a, b) in enumerate(voigt):
                for sign in (1.0, -1.0):
                    strain = np.zeros((3, 3))
                    strain[a, b] = sign * step
                    if a != b:
                        strain[b, a] = sign * step
                    moved = atoms.copy()
                    moved.set_cell(
                        cell @ (identity + strain), scale_atoms=False
                    )
                    moved.positions = positions @ (identity + strain)
                    self.clear_cache()
                    energy = self.energy(moved, method="analytic")
                    derivative[index] += sign * energy / (2.0 * step)
                if a != b:
                    derivative[index] *= 0.5
        finally:
            self.clear_cache()
        sigma = derivative / volume
        return np.array(
            [
                [sigma[0], sigma[5], sigma[4]],
                [sigma[5], sigma[1], sigma[3]],
                [sigma[4], sigma[3], sigma[2]],
            ]
        )
