"""Lennard-Jones tests against ASE and finite differences."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones as AseLennardJones

from arrayff.potentials import LennardJones, available_potentials, get_potential

SIGMA = 3.405
EPSILON = 0.01032
CUTOFF = 8.0


def _bulk():
    return bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))


def _cluster():
    return Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [0.0, 3.6, 0.0],
            [3.6, 3.6, 0.0],
        ],
        cell=None,
        pbc=False,
    )


def _arrayff(**kwargs):
    return LennardJones(
        sigma=SIGMA, epsilon=EPSILON, cutoff=CUTOFF, shift=True, **kwargs
    )


def test_energy_matches_ase():
    atoms = _bulk()
    calc = _arrayff()
    atoms.calc = AseLennardJones(sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF)
    assert calc.energy(atoms) == pytest.approx(
        atoms.get_potential_energy(), rel=1e-12
    )


def test_forces_match_ase():
    atoms = _bulk()
    calc = _arrayff()
    _, forces = calc.energy_and_forces(atoms)
    atoms.calc = AseLennardJones(sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF)
    np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-12)


def test_forces_match_finite_differences():
    atoms = _cluster()
    calc = _arrayff()
    _, forces = calc.energy_and_forces(atoms)
    step = 1e-6
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


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax", "autograd"])
def test_backends_agree(backend):
    pytest.importorskip(backend)
    if backend == "jax":
        from arrayff.utils import enable_x64

        enable_x64()
    atoms = _bulk()
    reference = _arrayff()
    energy_ref, forces_ref = reference.energy_and_forces(atoms)
    energy, forces = _arrayff(backend=backend).energy_and_forces(atoms)
    assert energy == pytest.approx(energy_ref, rel=1e-10)
    np.testing.assert_allclose(forces, forces_ref, atol=1e-9)


def test_shift_matches_pair_count():
    atoms = _cluster()
    plain = LennardJones(sigma=SIGMA, epsilon=EPSILON, cutoff=CUTOFF)
    shifted = _arrayff()
    from ase.neighborlist import neighbor_list

    n_pairs = len(neighbor_list("i", atoms, CUTOFF))
    value_at_cutoff = (
        4.0 * EPSILON * ((SIGMA / CUTOFF) ** 12 - (SIGMA / CUTOFF) ** 6)
    )
    delta = shifted.energy(atoms) - plain.energy(atoms)
    assert delta == pytest.approx(-0.5 * n_pairs * value_at_cutoff, rel=1e-12)


def test_registry():
    assert available_potentials()["lj"] is True
    calc = get_potential("lj", sigma=SIGMA, epsilon=EPSILON, cutoff=8.0)
    assert calc.cutoff == 8.0
    with pytest.raises(KeyError):
        get_potential("does-not-exist")
