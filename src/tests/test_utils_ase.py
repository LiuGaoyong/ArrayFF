"""Tests for the Atoms conversion and neighbour table helpers."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.neighborlist import neighbor_list as ase_neighbor_list

from arrayff.utils.ase import atoms_to_arrays, species_of
from arrayff.neighbors._table import neighbor_table


def _atoms():
    return bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))


def test_atoms_to_arrays_shapes():
    atoms = _atoms()
    positions, numbers, cell, pbc = atoms_to_arrays(atoms)
    assert positions.shape == (len(atoms), 3)
    assert numbers.shape == (len(atoms),)
    assert cell.shape == (3, 3)
    assert pbc.shape == (3,)
    np.testing.assert_allclose(positions, atoms.positions)
    np.testing.assert_array_equal(numbers, atoms.numbers)


def test_species_of():
    elements, species = species_of(np.array([18, 29, 18, 18]))
    np.testing.assert_array_equal(elements, [18, 29])
    np.testing.assert_array_equal(species, [0, 1, 0, 0])


def test_neighbor_table_matches_ase():
    atoms = _atoms()
    cutoff = 6.0
    table = neighbor_table(atoms, cutoff)
    i, j, shift, distance = ase_neighbor_list("ijSd", atoms, cutoff)

    assert int(table.mask.sum()) == len(i)
    assert table.index.shape == (len(atoms), int(np.bincount(i).max()))

    vectors = atoms.positions[table.index] - atoms.positions[:, None, :]
    vectors = vectors + table.shift @ atoms.cell.array
    padded = np.linalg.norm(vectors, axis=-1)[table.mask]
    np.testing.assert_allclose(np.sort(padded), np.sort(distance), atol=1e-12)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_neighbor_table_backend(backend):
    pytest.importorskip(backend)
    table = neighbor_table(_atoms(), 6.0, backend=backend)
    assert tuple(table.index.shape) == (len(_atoms()), table.max_neighbors)
    assert table.to_backend("numpy").index.shape == table.index.shape


def test_neighbor_table_max_neighbors_too_small():
    atoms = _atoms()
    with pytest.raises(ValueError, match="max_neighbors"):
        neighbor_table(atoms, 6.0, max_neighbors=1)


def test_empty_system():
    from ase import Atoms

    empty = Atoms()
    table = neighbor_table(empty, 6.0)
    assert table.index.shape == (0, 0)
    assert table.mask.shape == (0, 0)
