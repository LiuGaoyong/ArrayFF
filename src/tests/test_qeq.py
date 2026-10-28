"""QEq: charge conservation, symmetry, forces and backends."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from arrayff.electrostatics import QEq

PARAMS = dict(
    chi={8: 8.5, 1: 4.5},
    J={8: 10.0, 1: 12.0},
    shielding=1.0,
)


def _water():
    return Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])


def test_charge_conservation():
    charges = QEq(**PARAMS).charges(_water())
    assert charges.sum() == pytest.approx(0.0, abs=1e-10)


def test_symmetric_dimer_has_equal_charges():
    dimer = Atoms("H2", positions=[[0, 0, 0], [1.4, 0, 0]])
    charges = QEq(chi=4.5, J=12.0, shielding=1.0).charges(dimer)
    assert charges[0] == pytest.approx(charges[1], abs=1e-12)
    assert charges.sum() == pytest.approx(0.0, abs=1e-12)


def test_total_charge_constraint():
    charges = QEq(**PARAMS, q_total=1.0).charges(_water())
    assert charges.sum() == pytest.approx(1.0, abs=1e-10)


def test_forces_match_finite_differences():
    atoms = _water()
    calc = QEq(**PARAMS)
    _, forces = calc.energy_and_forces(atoms)
    step = 1e-5
    numeric = np.zeros_like(forces)
    for atom in range(len(atoms)):
        for axis in range(3):
            moved = atoms.copy()
            moved.positions[atom, axis] += step
            plus = calc.energy(moved)
            moved.positions[atom, axis] -= 2 * step
            minus = calc.energy(moved)
            numeric[atom, axis] = -(plus - minus) / (2 * step)
    np.testing.assert_allclose(forces, numeric, atol=1e-6)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_backends_agree(backend):
    if backend == "jax":
        from arrayff.utils import enable_x64

        enable_x64()
    pytest.importorskip(backend)
    atoms = _water()
    reference = QEq(**PARAMS).charges(atoms)
    charges = QEq(**PARAMS, backend=backend).charges(atoms)
    np.testing.assert_allclose(charges, reference, atol=1e-9)
