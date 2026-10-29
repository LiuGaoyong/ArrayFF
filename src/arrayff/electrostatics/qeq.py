"""Charge equilibration (QEq).

Implements the Rappe-Goddard charge equilibration model with the algebraic
shielded Coulomb kernel ``1 / sqrt(r^2 + lambda^2)`` and the extended matrix
formulation used by DMFF's QEq solver::

    [ H  1 ] [q   ]   [ -chi ]
    [ 1T 0 ] [lam ] = [  Q   ]

``H_ij = k_e / sqrt(r_ij^2 + lambda^2)`` for ``i != j`` and ``H_ii = J_i``.

Forces and stress are available in two charge-response modes (Q12=C):

* ``charge_response="frozen"``: Hellmann-Feynman truncation, the charges are
  solved once and detached (the historical behaviour).
* ``charge_response="relaxed"``: the charges are allowed to respond.  The AD
  path differentiates through the linear solve; the analytic path uses the
  envelope theorem (the first-order charge-response term vanishes at the QEq
  minimum), so the closed-form virial is the same as the frozen one to solver
  accuracy.  Tests compare both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from arrayff.potentials.base import PotentialBase
from arrayff.utils import ad
from arrayff.utils.ase import atoms_to_arrays, species_of
from arrayff.utils.dispatch import do, to_backend, to_numpy
from arrayff.utils.exceptions import ScopeError

__all__ = ["QEq"]

#: Coulomb constant in eV * Angstrom / e^2.
KE = 14.399645


class QEq(PotentialBase):
    """Charge equilibration for electronegativity/hardness parameters.

    Parameters
    ----------
    chi, J:
        Electronegativity (eV) and idempotential (eV/e^2), either a scalar or
        a mapping ``{atomic_number: value}``.
    shielding:
        Shielding length ``lambda`` in Angstrom.
    q_total:
        Total charge constraint.
    charge_response:
        ``"frozen"`` (default) or ``"relaxed"``.
    """

    supports_charges = True
    name = "qeq"

    def __init__(
        self,
        chi: float | dict[int, float],
        J: float | dict[int, float],
        shielding: float = 1.0,
        q_total: float = 0.0,
        *,
        charge_response: str = "frozen",
        backend: str = "numpy",
        dtype: Any = None,
        jit: bool = False,
        method: str = "analytic",
    ) -> None:
        super().__init__(
            backend=backend, dtype=dtype, jit=jit, method=method
        )
        if charge_response not in ("frozen", "relaxed"):
            raise ValueError(
                "charge_response must be 'frozen' or 'relaxed', got "
                f"{charge_response!r}"
            )
        self.chi = chi
        self.J = J
        self.shielding = float(shielding)
        self.q_total = float(q_total)
        self.charge_response = charge_response
        self._jit_cache: dict[tuple, Any] = {}

    def _per_element(self, value: float | dict[int, float], elements):
        if isinstance(value, dict):
            missing = [int(z) for z in elements if int(z) not in value]
            if missing:
                raise KeyError(
                    f"Missing QEq parameter for Z={missing}; "
                    f"available keys: {sorted(value)}"
                )
            return np.array([float(value[int(z)]) for z in elements])
        return np.full(len(elements), float(value))

    def _setup(self, atoms: Atoms):
        elements, species_np = species_of(atoms.numbers)
        positions, _, cell, _ = atoms_to_arrays(
            atoms, self.backend, self.dtype
        )
        chi = self._per_element(self.chi, elements)[species_np]
        hardness = self._per_element(self.J, elements)[species_np]
        pbc = np.asarray(atoms.pbc, dtype=bool)
        return (
            positions,
            cell,
            pbc,
            to_backend(chi, self.backend),
            to_backend(hardness, self.backend),
        )

    def _geometry(self, positions: Any, cell: Any, pbc: np.ndarray):
        vectors = positions[None, :, :] - positions[:, None, :]
        if pbc.any():
            pbc_b = to_backend(pbc.astype(bool), self.backend)
            inverse = do("linalg.inv", cell)
            fractional = do("einsum", "ijk,kl->ijl", vectors, inverse)
            fractional = fractional - do("floor", fractional + 0.5) * pbc_b
            vectors = do("einsum", "ijk,kl->ijl", fractional, cell)
        squared = do("sum", vectors * vectors, axis=-1)
        distance = do("sqrt", squared + self.shielding**2)
        return vectors, distance

    def _matrix(
        self,
        positions: Any,
        cell: Any,
        pbc: np.ndarray,
        chi: Any,
        hardness: Any,
    ):
        n = positions.shape[0]
        vectors, distance = self._geometry(positions, cell, pbc)
        eye = do("eye", n, like=positions)
        kernel = KE / distance * (1.0 - eye)
        matrix = kernel + do("diag", hardness)
        top = do(
            "concatenate",
            [matrix, do("ones", (n, 1), like=positions)],
            axis=1,
        )
        bottom = do(
            "concatenate",
            [
                do("ones", (1, n), like=positions),
                do("zeros", (1, 1), like=positions),
            ],
            axis=1,
        )
        augmented = do("concatenate", [top, bottom], axis=0)
        rhs = do(
            "concatenate",
            [
                -chi,
                do(
                    "reshape",
                    to_backend(self.q_total, self.backend),
                    (1,),
                ),
            ],
            axis=0,
        )
        solution = do("linalg.solve", augmented, rhs)
        return solution[:n], vectors, distance, kernel

    def _energy_from(
        self,
        charges: Any,
        chi: Any,
        hardness: Any,
        kernel: Any,
    ) -> Any:
        quadratic = do(
            "sum", (charges[:, None] * kernel) * charges[None, :]
        )
        onsite = do("sum", hardness * charges * charges)
        return do("sum", chi * charges) + 0.5 * (quadratic + onsite)

    def _analytic_terms(
        self,
        positions: Any,
        cell: Any,
        pbc: np.ndarray,
        chi: Any,
        hardness: Any,
        charges: Any,
        *,
        want_forces: bool,
        want_stress: bool,
    ):
        n = positions.shape[0]
        charges, vectors, distance, kernel = self._matrix(
            positions, cell, pbc, chi, hardness
        )
        energy = self._energy_from(charges, chi, hardness, kernel)
        forces = None
        stress = None
        if want_forces or want_stress:
            radius = do("sqrt", do("sum", vectors * vectors, axis=-1))
            radius_safe = do("where", radius > 0.0, radius, 1.0)
            eye = do("eye", n, like=positions)
            dK_dr = -KE * radius_safe / (distance**3)
            dE_dr = charges[:, None] * charges[None, :] * dK_dr
            coefficient = do("where", eye > 0.0, 0.0, dE_dr / radius_safe)
            if want_forces:
                # F_i = -dE/dr_i = +dE/dr * (r_j - r_i) / r
                forces = do("sum", coefficient[:, :, None] * vectors, axis=1)
            if want_stress:
                virial = 0.5 * do(
                    "einsum", "ij,ijk,ijl->kl", coefficient, vectors, vectors
                )
                volume = do("abs", do("linalg.det", cell))
                stress = virial / volume
        return energy, forces, stress

    def _stress_autodiff(
        self,
        positions: Any,
        cell: Any,
        pbc: np.ndarray,
        chi: Any,
        hardness: Any,
        charges: Any,
        volume: float,
    ) -> Any:
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
            if self.charge_response == "relaxed":
                q, _, _, kernel = self._matrix(
                    moved, deformed_cell, pbc, chi, hardness
                )
            else:
                q = charges
                _, _, _, kernel = self._matrix(
                    moved, deformed_cell, pbc, chi, hardness
                )
            return self._energy_from(q, chi, hardness, kernel)

        derivative = np.asarray(to_numpy(ad.jacrev(energy)(zero)), dtype=float)
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

    def clear_cache(self) -> None:
        self._jit_cache.clear()

    def _jit_evaluate(
        self,
        positions: Any,
        cell: Any,
        pbc: np.ndarray,
        chi: Any,
        hardness: Any,
        want: tuple[str, ...],
    ) -> dict[str, Any]:
        key = (
            "qeq",
            want,
            tuple(np.shape(positions)),
            tuple(np.shape(cell)),
            tuple(np.shape(chi)),
            tuple(np.shape(hardness)),
            tuple(bool(x) for x in pbc),
        )
        compiled = self._jit_cache.get(key)
        if compiled is None:

            def kernel(*args: Any) -> tuple[Any, ...]:
                pos, cel, chi_, hardness_ = args
                energy, forces, stress = self._analytic_terms(
                    pos,
                    cel,
                    pbc,
                    chi_,
                    hardness_,
                    None,
                    want_forces="forces" in want,
                    want_stress="stress" in want,
                )
                values = {"energy": energy}
                if forces is not None:
                    values["forces"] = forces
                if stress is not None:
                    values["stress"] = stress
                return tuple(values[name] for name in want)

            from arrayff.utils.jit import autojit

            compiled = autojit(kernel, backend=self.backend)
            self._jit_cache[key] = compiled
        values = compiled(positions, cell, chi, hardness)
        return dict(zip(want, values))

    def evaluate(
        self,
        atoms: Atoms,
        *,
        properties: tuple[str, ...] = ("energy",),
        method: str | None = None,
    ) -> dict[str, Any]:
        props = tuple(properties)
        unknown = set(props) - {"energy", "forces", "stress", "charges"}
        if unknown:
            raise ScopeError(
                f"Unsupported properties for QEq: {sorted(unknown)}"
            )
        if "stress" in props and atoms.cell.rank != 3:
            raise ValueError("stress requires a fully periodic cell")

        positions, cell, pbc, chi, hardness = self._setup(atoms)
        charges_now, _, _, _ = self._matrix(
            positions, cell, pbc, chi, hardness
        )
        resolved = self._resolve_method(method)
        volume = float(atoms.get_volume()) if "stress" in props else 1.0
        result: dict[str, Any] = {}

        if resolved == "analytic":
            numeric_want = tuple(p for p in props if p != "charges")
            if self.jit and numeric_want:
                try:
                    result.update(
                        self._jit_evaluate(
                            positions,
                            cell,
                            pbc,
                            chi,
                            hardness,
                            numeric_want,
                        )
                    )
                except Exception as exc:  # pragma: no cover - backend dependent
                    import warnings

                    warnings.warn(
                        "QEq jit failed; falling back to eager: " + str(exc),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._jit_cache.clear()
                    self.jit = False
                    energy, forces, stress = self._analytic_terms(
                        positions,
                        cell,
                        pbc,
                        chi,
                        hardness,
                        charges_now,
                        want_forces="forces" in props,
                        want_stress="stress" in props,
                    )
                    if "energy" in props or "forces" in props:
                        result["energy"] = energy
                    if forces is not None:
                        result["forces"] = forces
                    if stress is not None:
                        result["stress"] = stress
            else:
                energy, forces, stress = self._analytic_terms(
                    positions,
                    cell,
                    pbc,
                    chi,
                    hardness,
                    charges_now,
                    want_forces="forces" in props,
                    want_stress="stress" in props,
                )
                if "energy" in props or "forces" in props:
                    result["energy"] = energy
                if forces is not None:
                    result["forces"] = forces
                if stress is not None:
                    result["stress"] = stress
        elif resolved == "autodiff":
            if "energy" in props or "forces" in props:
                if self.charge_response == "relaxed":

                    def value(p: Any) -> Any:
                        q, _, _, kernel = self._matrix(
                            p, cell, pbc, chi, hardness
                        )
                        return self._energy_from(q, chi, hardness, kernel)

                else:
                    fixed = ad.stop_gradient(charges_now)

                    def value(p: Any) -> Any:
                        _, _, _, kernel = self._matrix(
                            p, cell, pbc, chi, hardness
                        )
                        return self._energy_from(fixed, chi, hardness, kernel)

                if "forces" in props:
                    energy, gradient = ad.value_and_grad(value)(positions)
                    result["energy"] = energy
                    result["forces"] = -gradient
                else:
                    result["energy"] = value(positions)
            if "stress" in props:
                result["stress"] = self._stress_autodiff(
                    positions,
                    cell,
                    pbc,
                    chi,
                    hardness,
                    charges_now,
                    volume,
                )
        else:
            energy, forces, _ = self._analytic_terms(
                positions,
                cell,
                pbc,
                chi,
                hardness,
                charges_now,
                want_forces="forces" in props,
                want_stress=False,
            )
            if "energy" in props or "forces" in props:
                result["energy"] = energy
            if forces is not None:
                result["forces"] = forces
            if "stress" in props:
                result["stress"] = self._stress_finite_difference(atoms)
        if "charges" in props:
            result["charges"] = to_numpy(charges_now)

        converted: dict[str, Any] = {}
        for key, value in result.items():
            if key == "energy":
                converted[key] = float(to_numpy(value))
            elif key == "charges":
                converted[key] = np.asarray(to_numpy(value))
            else:
                converted[key] = np.asarray(to_numpy(value))
        return converted

    def charges(self, atoms: Atoms) -> np.ndarray:
        """Return the equilibrium charges in units of the elementary charge."""
        positions, cell, pbc, chi, hardness = self._setup(atoms)
        charges, _, _, _ = self._matrix(positions, cell, pbc, chi, hardness)
        return np.asarray(to_numpy(charges))

    def _stress_finite_difference(self, atoms: Atoms, step: float = 1e-5):
        volume = float(atoms.get_volume())
        positions = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        identity = np.eye(3)
        voigt = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        derivative = np.zeros(6)
        for index, (a, b) in enumerate(voigt):
            for sign in (1.0, -1.0):
                strain = np.zeros((3, 3))
                strain[a, b] = sign * step
                if a != b:
                    strain[b, a] = sign * step
                moved = atoms.copy()
                moved.set_cell(cell @ (identity + strain), scale_atoms=False)
                moved.positions = positions @ (identity + strain)
                energy = self.energy(moved, method="analytic")
                derivative[index] += sign * energy / (2.0 * step)
            if a != b:
                derivative[index] *= 0.5
        sigma = derivative / volume
        return np.array(
            [
                [sigma[0], sigma[5], sigma[4]],
                [sigma[5], sigma[1], sigma[3]],
                [sigma[4], sigma[3], sigma[2]],
            ]
        )
