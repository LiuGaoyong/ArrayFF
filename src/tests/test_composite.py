"""Additive composite potentials."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from arrayff.electrostatics import QEq
from arrayff.potentials import LennardJones
from arrayff.potentials.composite import CompositePotential


def _lj():
    return LennardJones(cutoff=8.0, sigma=3.405, epsilon=0.01032)


def _water():
    return Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[6, 6, 6],
        pbc=True,
    )


def test_energy_and_forces_add():
    atoms = _water()
    lj = _lj()
    qeq = QEq(chi={8: 8.5, 1: 4.5}, J={8: 10.0, 1: 12.0})
    composite = lj + qeq
    assert isinstance(composite, CompositePotential)
    assert composite.supports_charges
    energy, forces = composite.energy_and_forces(atoms)
    ref_energy = lj.energy(atoms) + qeq.energy(atoms)
    ref_forces = (
        lj.energy_and_forces(atoms)[1] + qeq.energy_and_forces(atoms)[1]
    )
    assert energy == pytest.approx(ref_energy, rel=1e-12)
    np.testing.assert_allclose(forces, ref_forces, atol=1e-10)
    np.testing.assert_allclose(composite.charges(atoms), qeq.charges(atoms))
    np.testing.assert_allclose(
        composite.stress(atoms),
        lj.stress(atoms) + qeq.stress(atoms),
        atol=1e-10,
    )


def test_composite_flatten_and_sum():
    lj = _lj()
    qeq = QEq(chi=4.5, J=12.0)
    composite = lj + qeq + lj
    assert len(composite.components) == 3
    summed = sum([lj, qeq])
    assert isinstance(summed, CompositePotential)
    assert len(summed.components) == 2


def test_charge_component_constraint():
    q1 = QEq(chi=4.5, J=12.0)
    q2 = QEq(chi=4.5, J=12.0)
    with pytest.raises(ValueError, match="at most one charge"):
        _ = q1 + q2


def test_backend_mismatch_rejected():
    pytest.importorskip("torch")
    lj = _lj()
    qeq = QEq(chi=4.5, J=12.0, backend="torch")
    with pytest.raises(TypeError, match="backend"):
        _ = lj + qeq


def test_calculator_drives_composite():
    from arrayff.potentials import ArrayFFCalculator

    atoms = _water()
    atoms.calc = ArrayFFCalculator(
        _lj() + QEq(chi={8: 8.5, 1: 4.5}, J={8: 10.0, 1: 12.0})
    )
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert atoms.calc.results["charges"].shape == (3,)
    assert atoms.get_forces().shape == (3, 3)
