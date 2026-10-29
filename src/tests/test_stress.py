"""Stress tensor via finite differences of the energy."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT as AseEMT
from ase.calculators.lj import LennardJones as AseLJ

from arrayff.potentials import EMTCalculator
from arrayff.potentials import EMT, LennardJones


def _voigt(sigma):
    return np.array(
        [
            sigma[0, 0],
            sigma[1, 1],
            sigma[2, 2],
            sigma[1, 2],
            sigma[0, 2],
            sigma[0, 1],
        ]
    )


def test_emt_stress_matches_ase():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    sigma = EMT().stress(atoms)
    atoms.calc = AseEMT()
    np.testing.assert_allclose(_voigt(sigma), atoms.get_stress(), atol=1e-8)


def test_lj_stress_matches_ase():
    atoms = bulk("Ar", "fcc", a=5.2, cubic=True).repeat((2, 2, 2))
    calc = LennardJones(cutoff=8.0, sigma=3.405, epsilon=0.01032, shift=True)
    sigma = calc.stress(atoms)
    atoms.calc = AseLJ(sigma=3.405, epsilon=0.01032, rc=8.0)
    np.testing.assert_allclose(_voigt(sigma), atoms.get_stress(), atol=1e-8)


def test_diagonal_stress_is_isotropic_for_cubic_crystal():
    atoms = bulk("Cu", "fcc", a=3.5, cubic=True).repeat((2, 2, 2))
    sigma = EMT().stress(atoms)
    assert sigma[0, 0] == pytest.approx(sigma[1, 1], abs=1e-9)
    assert sigma[0, 0] == pytest.approx(sigma[2, 2], abs=1e-9)
    atoms.calc = AseEMT()
    np.testing.assert_allclose(_voigt(sigma), atoms.get_stress(), atol=1e-8)


def test_calculator_exposes_stress():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    atoms.calc = EMTCalculator()
    stress = atoms.get_stress()
    assert stress.shape == (6,)
    assert abs(stress[0] - stress[1]) < 1e-9


def test_stress_requires_periodic_cell():
    from ase import Atoms

    molecule = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    with pytest.raises(ValueError, match="periodic"):
        LennardJones(cutoff=8.0, sigma=3.4, epsilon=0.01).stress(molecule)
