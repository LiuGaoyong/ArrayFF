"""Effective medium theory: agreement with ASE and backend consistency."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT as AseEMT

from arrayff.potentials import EMT

METALS = [
    ("Cu", 3.6),
    ("Ag", 4.09),
    ("Au", 4.08),
    ("Ni", 3.52),
    ("Al", 4.05),
    ("Pd", 3.89),
    ("Pt", 3.92),
]


@pytest.mark.parametrize("symbol,a", METALS)
def test_energy_and_forces_match_ase(symbol, a):
    atoms = bulk(symbol, "fcc", a=a, cubic=True).repeat((2, 2, 2))
    energy, forces = EMT().energy_and_forces(atoms)
    atoms.calc = AseEMT()
    assert energy == pytest.approx(atoms.get_potential_energy(), abs=1e-11)
    np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-11)


def test_isolated_atom_energy():
    from ase.calculators.emt import EMT as BaseEMT

    atom = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    energy, forces = EMT().energy_and_forces(atom)
    atom.calc = BaseEMT()
    assert energy == pytest.approx(atom.get_potential_energy(), abs=1e-12)
    np.testing.assert_allclose(forces, 0.0, atol=1e-12)


def test_surface_relaxation_force_is_nonzero():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    atoms[0].position[2] += 0.2
    _, forces = EMT().energy_and_forces(atoms)
    assert np.abs(forces[0]).max() > 1e-3


@pytest.mark.parametrize("backend", ["torch", "jax", "autograd"])
def test_backends_agree(backend):
    pytest.importorskip(backend)
    if backend == "jax":
        from arrayff.utils import enable_x64

        enable_x64()
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    reference = EMT().energy_and_forces(atoms)
    energy, forces = EMT(backend=backend).energy_and_forces(atoms)
    assert energy == pytest.approx(reference[0], rel=1e-10)
    np.testing.assert_allclose(forces, reference[1], atol=1e-9)


def test_unsupported_element():
    atoms = Atoms("Xe", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(NotImplementedError, match="Xe"):
        EMT().energy(atoms)
