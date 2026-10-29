"""ASE calculator adapters."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT as AseEMT

from arrayff.electrostatics import QEqCalculator
from arrayff.potentials import EMTCalculator, LJCalculator
from arrayff.potentials import EMT, LennardJones


def test_lj_calculator_matches_potential():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    reference = LennardJones()
    atoms.calc = LJCalculator()
    assert atoms.get_potential_energy() == pytest.approx(
        reference.energy(atoms), rel=1e-12
    )
    np.testing.assert_allclose(
        atoms.get_forces(), reference.energy_and_forces(atoms)[1], atol=1e-12
    )


def test_emt_calculator_matches_ase():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    atoms.calc = EMTCalculator()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy == pytest.approx(EMT().energy(atoms), abs=1e-12)
    atoms.calc = AseEMT()
    assert energy == pytest.approx(atoms.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-10)


def test_qeq_calculator_exposes_charges():
    from ase import Atoms

    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = QEqCalculator(chi={8: 8.5, 1: 4.5}, J={8: 10.0, 1: 12.0})
    atoms.get_potential_energy()
    charges = atoms.calc.results["charges"]
    assert charges.shape == (3,)
    assert charges.sum() == pytest.approx(0.0, abs=1e-10)


def test_calculator_runs_optimizer_step():
    from ase.optimize import FIRE

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.rattle(0.05, seed=1)
    atoms.calc = EMTCalculator()
    initial = atoms.get_potential_energy()
    FIRE(atoms, logfile=None).run(steps=3)
    assert atoms.get_potential_energy() < initial


def test_calculator_properties_are_dynamic():
    from ase.build import bulk

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    atoms.calc = LJCalculator()
    assert "charges" not in atoms.calc.implemented_properties
    atoms.get_potential_energy()
    assert "forces" not in atoms.calc.results
