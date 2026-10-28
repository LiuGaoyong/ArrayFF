"""Neighbour-table caching and alternative builders."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from arrayff.potentials import EMT
from arrayff.utils.neighbors import neighbor_table


def _atoms():
    return bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))


def test_table_is_cached_for_unchanged_geometry():
    atoms = _atoms()
    calc = EMT()
    first = calc._table(atoms)
    second = calc._table(atoms)
    assert first is second


def test_table_is_rebuilt_after_displacement():
    atoms = _atoms()
    calc = EMT()
    first = calc._table(atoms)
    atoms.positions[0, 0] += 0.05
    second = calc._table(atoms)
    assert second is not first


def test_clear_cache():
    atoms = _atoms()
    calc = EMT()
    first = calc._table(atoms)
    calc.clear_cache()
    assert calc._table(atoms) is not first


@pytest.mark.parametrize("builder", ["vesin", "matscipy"])
def test_alternative_builders_match_sklearn(builder):
    pytest.importorskip(builder)
    atoms = _atoms()
    reference = neighbor_table(atoms, 6.0, builder="sklearn")
    other = neighbor_table(atoms, 6.0, builder=builder)
    assert other.index.shape == reference.index.shape
    assert int(other.mask.sum()) == int(reference.mask.sum())

    def distances(table):
        vectors = atoms.positions[table.index] - atoms.positions[:, None, :]
        vectors = vectors + table.shift @ atoms.cell.array
        return np.sort(np.linalg.norm(vectors, axis=-1)[table.mask])

    np.testing.assert_allclose(
        distances(other), distances(reference), atol=1e-12
    )


def test_potential_with_vesin_builder_matches_default():
    pytest.importorskip("vesin")
    atoms = _atoms()
    reference = EMT().energy(atoms)
    assert EMT(builder="vesin").energy(atoms) == pytest.approx(
        reference, rel=1e-12
    )


def test_skin_does_not_change_the_energy():
    atoms = _atoms()
    for cls in (EMT,):
        reference = cls().energy(atoms)
        assert cls(skin=0.6).energy(atoms) == pytest.approx(
            reference, rel=1e-12
        )


def test_skin_reuses_table_for_small_moves():
    atoms = _atoms()
    calc = EMT(skin=0.6)
    first = calc._table(atoms)
    atoms.positions[0, 0] += 0.05  # below skin / 2
    assert calc._table(atoms) is first


def test_skin_rebuilds_after_large_move():
    atoms = _atoms()
    calc = EMT(skin=0.6)
    first = calc._table(atoms)
    atoms.positions[0, 0] += 0.5  # above skin / 2
    assert calc._table(atoms) is not first


def test_skin_with_kim_lj_per_pair_cutoff():
    pytest.importorskip("ase")
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    from arrayff.potentials import LennardJones

    reference = LennardJones().energy(atoms)
    assert LennardJones(skin=0.7).energy(atoms) == pytest.approx(
        reference, rel=1e-12
    )
