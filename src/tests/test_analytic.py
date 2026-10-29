"""Analytic forces/stress must agree with autodiff and reference values."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT as AseEMT

from arrayff.electrostatics import QEq
from arrayff.potentials import EMT, LennardJones


def _ar():
    return bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))


def _cu():
    return bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))


@pytest.mark.parametrize("layout", ["edges", "padded"])
@pytest.mark.parametrize("smooth", [False, True])
def test_lj_analytic_matches_autodiff(layout, smooth):
    atoms = _ar()
    calc = LennardJones(
        cutoff=8.0,
        sigma=3.405,
        epsilon=0.01032,
        shift=True,
        smooth=smooth,
        layout=layout,
    )
    analytic = calc.energy_and_forces(atoms, method="analytic")
    autodiff = calc.energy_and_forces(atoms, method="autodiff")
    assert analytic[0] == pytest.approx(autodiff[0], rel=1e-12)
    np.testing.assert_allclose(analytic[1], autodiff[1], atol=1e-10)
    np.testing.assert_allclose(
        calc.stress(atoms, method="analytic"),
        calc.stress(atoms, method="autodiff"),
        atol=1e-10,
    )


@pytest.mark.parametrize("asap_cutoff", [False, True])
def test_emt_analytic_matches_autodiff_and_ase(asap_cutoff):
    atoms = _cu()
    calc = EMT(asap_cutoff=asap_cutoff)
    analytic = calc.energy_and_forces(atoms, method="analytic")
    autodiff = calc.energy_and_forces(atoms, method="autodiff")
    assert analytic[0] == pytest.approx(autodiff[0], rel=1e-12)
    np.testing.assert_allclose(analytic[1], autodiff[1], atol=1e-10)
    np.testing.assert_allclose(
        calc.stress(atoms, method="analytic"),
        calc.stress(atoms, method="autodiff"),
        atol=1e-10,
    )
    atoms.calc = AseEMT(asap_cutoff=asap_cutoff)
    assert analytic[0] == pytest.approx(
        atoms.get_potential_energy(), abs=1e-11
    )
    np.testing.assert_allclose(analytic[1], atoms.get_forces(), atol=1e-11)


def test_qeq_frozen_relaxed_and_stress():
    atoms = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[6, 6, 6],
        pbc=True,
    )
    params = dict(chi={8: 8.5, 1: 4.5}, J={8: 10.0, 1: 12.0})
    for response in ("frozen", "relaxed"):
        calc = QEq(**params, charge_response=response)
        analytic = calc.energy_and_forces(atoms, method="analytic")
        autodiff = calc.energy_and_forces(atoms, method="autodiff")
        assert analytic[0] == pytest.approx(autodiff[0], rel=1e-12)
        np.testing.assert_allclose(analytic[1], autodiff[1], atol=1e-8)
        np.testing.assert_allclose(
            calc.stress(atoms, method="analytic"),
            calc.stress(atoms, method="autodiff"),
            atol=1e-8,
        )


def test_qeq_mixed_pbc():
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [1.0, 0, 0]],
        cell=[3, 10, 10],
        pbc=[True, False, False],
    )
    calc = QEq(chi=4.5, J=12.0)
    charges = calc.charges(atoms)
    assert charges.shape == (2,)
    assert charges.sum() == pytest.approx(0.0, abs=1e-10)
