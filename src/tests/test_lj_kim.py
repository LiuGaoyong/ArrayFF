"""OpenKIM universal shifted Lennard-Jones parameters."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones as AseLennardJones

from arrayff.potentials import LennardJones
from arrayff.potentials.lj_kim import KIM_LJ_PARAMETERS


def _atoms(symbol="Ar", a=5.26):
    return bulk(symbol, "fcc", a=a, cubic=True).repeat((2, 2, 2))


def test_table_is_complete():
    assert set(KIM_LJ_PARAMETERS) == set(range(1, 119))
    cutoff, epsilon, sigma = KIM_LJ_PARAMETERS[18]
    assert cutoff == pytest.approx(7.55482)
    assert epsilon == pytest.approx(0.0123529)
    assert sigma == pytest.approx(1.88871)


def test_kim_energy_matches_ase_at_four_sigma():
    atoms = _atoms()
    cutoff, epsilon, sigma = KIM_LJ_PARAMETERS[18]
    energy = LennardJones().energy(atoms)
    atoms.calc = AseLennardJones(sigma=sigma, epsilon=epsilon, rc=cutoff)
    assert energy == pytest.approx(atoms.get_potential_energy(), rel=1e-12)


def test_kim_forces_match_ase():
    atoms = _atoms()
    cutoff, epsilon, sigma = KIM_LJ_PARAMETERS[18]
    _, forces = LennardJones().energy_and_forces(atoms)
    atoms.calc = AseLennardJones(sigma=sigma, epsilon=epsilon, rc=cutoff)
    np.testing.assert_allclose(forces, atoms.get_forces(), atol=1e-12)


def test_kim_rejects_explicit_cutoff():
    with pytest.raises(ValueError, match="4\\*sigma"):
        LennardJones(cutoff=8.0)


def test_mixed_species_uses_lorentz_berthelot():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    energy = LennardJones().energy(atoms)
    assert np.isfinite(energy)
    # Same structure with the KIM parameters of copper must be reproducible
    # from an explicit global-cutoff run at the pair cutoff.
    explicit = LennardJones(
        sigma=KIM_LJ_PARAMETERS[29][2],
        epsilon=KIM_LJ_PARAMETERS[29][1],
        cutoff=KIM_LJ_PARAMETERS[29][0],
        shift=True,
    )
    assert energy == pytest.approx(explicit.energy(atoms), rel=1e-12)
