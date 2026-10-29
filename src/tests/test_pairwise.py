"""Generic pairwise layer: forms, layout equivalence and new potentials."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from arrayff.potentials import LJ93, LennardJones, Morse, get_potential
from arrayff.potentials.pairwise.forms import GeneralizedLJ, Morse as MorseForm

R = np.linspace(2.0, 6.0, 400)


def _params(r0, depth):
    return {
        "r0": np.full_like(R, r0),
        "depth": np.full_like(R, depth),
    }


def test_generalized_lj_has_minimum_at_r0():
    form = GeneralizedLJ(12, 6)
    values = form.energy(R, _params(3.5, 0.12))
    index = int(np.argmin(values))
    assert R[index] == pytest.approx(3.5, abs=0.05)
    assert values[index] == pytest.approx(-0.12, rel=1e-3)


def test_generalized_lj_reproduces_the_usual_12_6_form():
    sigma, epsilon = 3.405, 0.01032
    r0 = 2.0 ** (1.0 / 6.0) * sigma
    form = GeneralizedLJ(12, 6)
    value = form.energy(R, _params(r0, epsilon))
    reference = 4.0 * epsilon * ((sigma / R) ** 12 - (sigma / R) ** 6)
    np.testing.assert_allclose(value, reference, rtol=1e-12)


def test_lj93_minimum():
    form = GeneralizedLJ(9, 3)
    values = form.energy(R, _params(3.2, 0.2))
    index = int(np.argmin(values))
    assert R[index] == pytest.approx(3.2, abs=0.05)
    assert values[index] == pytest.approx(-0.2, rel=1e-3)


def test_morse_form_minimum():
    form = MorseForm()
    r0, depth, a = 3.8, 0.01, 1.5
    params = {
        "r0": np.full_like(R, r0),
        "depth": np.full_like(R, depth),
        "a": np.full_like(R, a),
    }
    values = form.energy(R, params)
    index = int(np.argmin(values))
    assert R[index] == pytest.approx(r0, abs=0.05)
    assert values[index] == pytest.approx(-depth, rel=1e-3)


def test_morse_energy_and_forces_finite_differences():
    atoms = Atoms(
        "Ar4",
        positions=[[0, 0, 0], [3.8, 0, 0], [0, 3.9, 0], [3.7, 3.8, 0]],
        cell=None,
        pbc=False,
    )
    calc = Morse(r0=3.8, depth=0.01, a=1.5, cutoff=10.0)
    energy, forces = calc.energy_and_forces(atoms)
    assert np.isfinite(energy)
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
    np.testing.assert_allclose(forces, numeric, atol=1e-7)


def test_lj93_potential_runs():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    calc = LJ93(r0=3.8, depth=0.01, cutoff=10.0)
    assert np.isfinite(calc.energy(atoms))


def test_edges_and_padded_layouts_agree():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    edges = LennardJones()
    padded = LennardJones(layout="padded")
    assert padded.energy(atoms) == pytest.approx(edges.energy(atoms), rel=1e-12)
    np.testing.assert_allclose(
        padded.energy_and_forces(atoms)[1],
        edges.energy_and_forces(atoms)[1],
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_morse_backends_agree(backend):
    if backend == "jax":
        from arrayff.utils import enable_x64

        enable_x64()
    pytest.importorskip(backend)
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    reference = Morse(r0=3.8, depth=0.01, a=1.5).energy_and_forces(atoms)
    other = Morse(r0=3.8, depth=0.01, a=1.5, backend=backend).energy_and_forces(
        atoms
    )
    assert other[0] == pytest.approx(reference[0], rel=1e-10)
    np.testing.assert_allclose(other[1], reference[1], atol=1e-9)


def test_potentials_are_registered():
    assert get_potential("morse", r0=3.8, depth=0.01).name == "morse"
    assert get_potential("lj93", r0=3.8, depth=0.01).name == "lj93"


def test_morse_torch_matches_morse():
    torch = pytest.importorskip("torch")
    from arrayff.nn import MorseTorch

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    model = MorseTorch(
        species=[18], initial={18: {"r0": 3.8, "depth": 0.01, "a": 1.5}}
    )
    reference = Morse(r0=3.8, depth=0.01, a=1.5).energy(atoms)
    assert model(atoms).detach().item() == pytest.approx(reference, rel=1e-12)

    loss = (model(atoms) - (-1.0)) ** 2
    loss.backward()
    assert model.raw["r0"].grad is not None
    _ = torch
