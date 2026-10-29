"""EAM: agreement with ASE on a generated setfl file."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.eam import EAM as AseEAM

from arrayff.potentials import EAM


def _write_table(path: pathlib.Path) -> pathlib.Path:
    nrho, drho, nr, dr, cutoff = 2000, 0.01, 6000, 0.001, 6.0
    rho = np.arange(nrho) * drho
    r = np.arange(nr) * dr
    embedded = -np.sqrt(rho) + 0.1 * rho
    density = 10.0 * np.clip(1.0 - r / cutoff, 0.0, None) ** 4
    rphi = np.zeros(nr)
    rphi[1:] = 2.0 * (2.5 / r[1:]) ** 6 - 0.5 * (2.5 / r[1:]) ** 3
    with open(path, "w") as handle:
        handle.write("synthetic Cu\n\narrayff test\n1\nCu\n")
        handle.write(f"{nrho} {drho} {nr} {dr} {cutoff}\n")
        handle.write("29 63.546 3.615 FCC\n")
        for values in (embedded, density, rphi):
            handle.write("\n".join(f"{v:.10e}" for v in values) + "\n")
    return path


@pytest.fixture(scope="module")
def table_file(tmp_path_factory):
    return _write_table(tmp_path_factory.mktemp("eam") / "Cu.eam.alloy")


def _atoms():
    return bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))


def test_energy_and_forces_match_ase(table_file):
    atoms = _atoms()
    energy, forces = EAM(table_file).energy_and_forces(atoms)
    atoms.calc = AseEAM(potential=str(table_file))
    # Linear interpolation against ASE's cubic spline: 1e-4 relative energy.
    assert energy == pytest.approx(atoms.get_potential_energy(), rel=1e-4)
    np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-8)


@pytest.mark.parametrize("backend", ["torch", "jax", "advect"])
def test_backends_agree(table_file, backend):
    pytest.importorskip(backend)
    if backend == "jax":
        from arrayff.utils import enable_x64

        enable_x64()
    atoms = _atoms()
    energy_ref, forces_ref = EAM(table_file).energy_and_forces(atoms)
    energy, forces = EAM(table_file, backend=backend).energy_and_forces(atoms)
    assert energy == pytest.approx(energy_ref, rel=1e-8)
    np.testing.assert_allclose(forces, forces_ref, atol=1e-8)


def test_forces_match_finite_differences(table_file):
    atoms = _atoms()
    calc = EAM(table_file)
    _, forces = calc.energy_and_forces(atoms)
    step = 1e-6
    numeric = np.zeros_like(forces)
    for atom in range(2):
        for axis in range(3):
            moved = atoms.copy()
            moved.positions[atom, axis] += step
            plus = calc.energy(moved)
            moved.positions[atom, axis] -= 2 * step
            minus = calc.energy(moved)
            numeric[atom, axis] = -(plus - minus) / (2 * step)
    np.testing.assert_allclose(forces[:2], numeric[:2], atol=1e-5)


def test_unknown_element(table_file):
    atoms = bulk("Ag", "fcc", a=4.09, cubic=True)
    with pytest.raises(KeyError, match="Z=47"):
        EAM(table_file).energy(atoms)


def test_analytic_matches_autodiff_and_stress(table_file):
    atoms = _atoms()
    calc = EAM(table_file)
    analytic = calc.energy_and_forces(atoms, method="analytic")
    autodiff = calc.energy_and_forces(atoms, method="autodiff")
    assert analytic[0] == pytest.approx(autodiff[0], rel=1e-12)
    np.testing.assert_allclose(analytic[1], autodiff[1], atol=1e-8)
    np.testing.assert_allclose(
        calc.stress(atoms, method="analytic"),
        calc.stress(atoms, method="autodiff"),
        atol=1e-8,
    )
