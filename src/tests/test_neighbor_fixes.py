"""Regression tests for neighbour-list correctness fixes."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from arrayff.neighbors._list import neighbor_list
from arrayff.neighbors._table import NeighborPaddedTable


def test_non_periodic_cell_does_not_use_images():
    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3) * 3, pbc=False
    )
    for backend in ("sklearn", "pymatgen"):
        pytest.importorskip(backend if backend == "sklearn" else "pymatgen")
        i, j, shift = neighbor_list("ijS", atoms, 4.0, backend=backend)
        assert len(i) == 2
        np.testing.assert_array_equal(shift, 0)


def test_single_atom_returns_requested_quantities():
    atoms = Atoms("H", positions=[[0, 0, 0]])
    result = neighbor_list("d", atoms, 2.0, backend="sklearn")
    assert len(result) == 1
    assert result[0].shape == (0,)


def test_indices_are_mapped_to_original_atoms():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    for backend in ("sklearn", "pymatgen"):
        pytest.importorskip(backend if backend == "sklearn" else "pymatgen")
        i, _, _ = neighbor_list(
            "ijS", atoms, 4.0, indices=[5], backend=backend
        )
        np.testing.assert_array_equal(np.unique(i), [5])


def test_sorted_by_first_atom():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    i, _, _ = neighbor_list("ijS", atoms, 4.0, backend="sklearn")
    assert np.all(i[:-1] <= i[1:])


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_empty_table_stays_on_backend(backend):
    pytest.importorskip(backend)
    table = NeighborPaddedTable.from_ase(Atoms(), 6.0, backend=backend)
    module = type(table.index).__module__
    assert (backend in module) or (
        backend == "jax" and "jaxlib" in module
    )
    assert table.index.shape == (0, 0)
    assert table.n_neighbors_total == 0


def test_neighbor_count_is_backend_independent():
    torch = pytest.importorskip("torch")
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    numpy_table = NeighborPaddedTable.from_ase(atoms, 6.0)
    torch_table = NeighborPaddedTable.from_ase(atoms, 6.0, backend="torch")
    assert torch_table.n_neighbors_total == numpy_table.n_neighbors_total
    _ = torch
