import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from arrayff.neighborlist import (
    _ase_neighbor_list_3loop,
    _ase_neighbor_list_kdtree,
    _sklearn_query_ball_point,
    distance_pairs,
    neighbor_list,
)

_LST_2 = ["vesin", "matscipy"]
_LST_1 = ["sklearn", "pymatgen"]


def test_sklearn_query_ball_point() -> None:
    arr0 = np.random.rand(10, 3)
    arr1 = np.random.rand(3, 3)
    max_distance = 0.5
    i, j, _ = _sklearn_query_ball_point(arr0, arr1, max_distance)
    print("", i, j, sep="\n")
    assert i.max() < len(arr0)
    assert j.max() < len(arr1)


@pytest.mark.parametrize("backend", _LST_1)
def test_distance_pairs_atoms(backend: str) -> None:
    atoms = fcc111(
        "Cu",
        [6, 8, 1],
        vacuum=10,
        orthogonal=True,
        periodic=True,
    )
    p2 = atoms.positions.copy()
    cell = atoms.cell.array.copy()

    p1 = p2[:2, :]
    i, j, S, D, d = distance_pairs("ijSDd", p1, p2, cell, 3, backend)
    for k in range(len(S)):
        print(p1[i[k]], p2[j[k]], sep="\n")
        print(S[k], D[k], d[k])
        print()
    np.testing.assert_array_almost_equal(d, np.linalg.norm(D, axis=1))


@pytest.mark.parametrize("backend", _LST_2 + _LST_1)
class TestNeighborList:
    @staticmethod
    def print_result(result: tuple[np.ndarray, ...]) -> None:
        lst: list[np.ndarray] = []
        for arr in result:
            if arr.ndim == 1:
                lst.append(arr)
            elif arr.ndim == 2:
                for i in range(arr.shape[1]):
                    lst.append(arr[:, i])
        print(*lst, sep="\n")

    @staticmethod
    def check_result(
        result: dict[str, np.ndarray],
        other: dict[str, np.ndarray],
        bothways: bool = True,
    ) -> None:
        assert set(result.keys()) >= set("ijS")
        assert set(other.keys()) >= set("ijS")
        if not bothways:
            result["i"], result["j"], result["S"] = (
                np.append(result["i"], result["j"]),
                np.append(result["j"], result["i"]),
                np.vstack([result["S"], -result["S"]]),
            )
            other["i"], other["j"], other["S"] = (
                np.append(other["i"], other["j"]),
                np.append(other["j"], other["i"]),
                np.vstack([other["S"], -other["S"]]),
            )
        i0, j0, (Sx0, Sy0, Sz0) = result["i"], result["j"], result["S"].T
        i1, j1, (Sx1, Sy1, Sz1) = other["i"], other["j"], other["S"].T
        idx0 = np.lexsort((Sz0, Sy0, Sx0, j0, i0))
        idx1 = np.lexsort((Sz1, Sy1, Sx1, j1, i1))
        arr_1 = np.column_stack((i1, j1, Sx1, Sy1, Sz1))[idx1].astype(int)
        arr_0 = np.column_stack((i0, j0, Sx0, Sy0, Sz0))[idx0].astype(int)
        print("-" * 56, arr_0.T, arr_1.T, arr_0.T - arr_1.T, "-" * 56, sep="\n")
        np.testing.assert_array_equal(arr_0, arr_1)

    @classmethod
    def _test_neighbor_list_for_atoms(
        cls,
        atoms: Atoms,
        backend: str,
        cutoff: float = 3.0,
        idx: list[int] | None = None,
    ) -> None:
        quantities = "ijSd"
        print()
        print("#############################################################")
        print(f"Test neighbor list for backend: {backend}")
        print("=============================================================")
        print(atoms)
        if backend not in _LST_2:
            print(f"Neighbor list with indices: {idx}")
            res_1 = neighbor_list(
                "ijSd",
                atoms,
                cutoff,
                indices=idx,
                backend="ase_kdtree",
            )
            res_0 = neighbor_list(
                "ijSd",
                atoms,
                cutoff,
                indices=idx,
                backend=backend,
            )
            print("standard")
            cls.print_result(res_1)
            print(backend)
            cls.print_result(res_0)
            cls.check_result(
                {k: res_0[i] for i, k in enumerate(quantities)},
                {k: res_1[i] for i, k in enumerate(quantities)},
                bothways=True,
            )
        print("=============================================================")
        print("Full neighbor list")
        res_0 = _ase_neighbor_list_3loop(quantities, atoms, cutoff)
        res_1 = neighbor_list(
            quantities,
            atoms,
            cutoff,
            bothways=True,
            backend=backend,
        )
        print("standard")
        cls.print_result(res_0)
        print(backend)
        cls.print_result(res_1)
        cls.check_result(
            {k: res_0[i] for i, k in enumerate(quantities)},
            {k: res_1[i] for i, k in enumerate(quantities)},
            bothways=True,
        )
        print("=============================================================")
        print("Half neighbor list")
        res_0 = _ase_neighbor_list_kdtree(
            quantities,
            atoms,
            cutoff,
            bothways=False,
        )
        res_1 = neighbor_list(
            quantities,
            atoms,
            cutoff,
            bothways=False,
            backend=backend,
        )
        print("standard")
        cls.print_result(res_0)
        print(backend)
        cls.print_result(res_1)
        cls.check_result(
            {k: res_0[i] for i, k in enumerate(quantities)},
            {k: res_1[i] for i, k in enumerate(quantities)},
            bothways=False,
        )

    def test_neighbor_list_for_atoms_nopbc(self, backend: str) -> None:
        self._test_neighbor_list_for_atoms(
            atoms=molecule("CH4"),
            backend=str(backend).lower(),
            cutoff=1.5,
            idx=[0],
        )

    def test_neighbor_list_for_atoms_pbc(self, backend: str) -> None:
        self._test_neighbor_list_for_atoms(
            atoms=fcc111(
                "Cu",
                [2, 2, 1],
                vacuum=10,
                orthogonal=True,
                periodic=True,
            ),
            backend=str(backend).lower(),
            cutoff=3.0,
            idx=[0, 1],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
