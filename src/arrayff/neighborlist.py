import itertools
from typing import Literal

import numpy as np
from ase.atoms import Atoms
from ase.cell import Cell
from ase.geometry import minkowski_reduce
from ase.neighborlist import _calc_expansion as __ase_calc_expansion
from ase.neighborlist import build_neighbor_list as __build_neighbor_list_ase
from ase.neighborlist import neighbor_list as _ase_neighbor_list_3loop
from numpy.typing import ArrayLike
from scipy import sparse as sp
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "neighbor_list",
    "distance_pairs",
]


#############################################################
# sklearn/scipy for `distance_pairs`
def _sklearn_query_ball_point(
    p1: np.ndarray,
    p2: np.ndarray | None = None,
    max_distance: float = float("inf"),
    return_distance: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Query ball point by distance.

    Args:
        p1 (np.ndarray): The query point or points.
        p2 (np.ndarray): The query point or points.
        max_distance (float): Maximum distance of neighborhoods.
        return_distance (bool, optional): Whether to return distance.
    """
    p1 = np.asarray(p1, dtype=float)
    if p2 is None:
        p2_: np.ndarray = p1
    else:
        p2_ = np.asarray(p2, dtype=float)
    assert isinstance(p2_, np.ndarray)
    l1, l2 = p1.shape[0], p2_.shape[0]
    if l1 >= l2:
        swap = False
    else:
        swap, p1, p2_ = True, p2_, p1
    assert p1.shape[0] >= p2_.shape[0]

    nn = NearestNeighbors()
    nn.fit(X=p1, y=None)
    distance = nn.radius_neighbors_graph(
        X=p2_,
        radius=float(max_distance),
        mode="distance" if return_distance else "connectivity",
    )
    m = sp.coo_matrix(distance if swap else distance.T)
    d = distance.data if return_distance else None
    assert m.shape == (l1, l2), (
        f"Invalid shape: {m.shape} "  #
        f"(expect {l1, l2})"
    )
    return m.row, m.col, d


def _sklearn_get_distance_pairs(
    p1: np.ndarray,
    p2: np.ndarray | None = None,
    cell: np.ndarray | Cell | None = None,
    max_distance: float = float("inf"),
    return_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(cell, Cell):
        cell = Cell.new(cell=cell)
    if cell.rank == 0:
        source, target, dist = _sklearn_query_ball_point(
            p1=p1,
            p2=p2,
            max_distance=max_distance,
            return_distance=bool(return_distance),
        )
        shift = np.zeros((len(source), 3), int)
        return source, target, shift, dist
    else:
        assert np.isfinite(max_distance), (
            "max_distance must be finite for periodic cell."
        )
        # The Minkowski-reduced cell is used for the periodic expansion,
        # while the returned shift vectors are expressed in the original cell.
        rcell, op = minkowski_reduce(cell)
        rcell = np.asarray(rcell, dtype=float)
        op = np.asarray(op, dtype=int)
        p1 = np.asarray(p1, dtype=float)
        if p2 is None:
            p2 = p1
        else:
            p2 = np.asarray(p2, dtype=float)
        l1 = p1.shape[0]
        # Wrap the points into the reduced cell and record the integer cell
        # offsets, i.e. p_wrapped = p + offset @ rcell.
        inv_rcell = np.linalg.pinv(rcell)
        offset1 = -np.floor(p1 @ inv_rcell).astype(int)
        offset2 = -np.floor(p2 @ inv_rcell).astype(int)
        p1_wrapped = p1 + offset1 @ rcell
        p2_wrapped = p2 + offset2 @ rcell
        # Enumerate all the periodic images of the first points set.
        pbc = np.asarray([True, True, True])
        n = __ase_calc_expansion(rcell, pbc, max_distance / 2)
        nrange3 = [range(-i, i + 1) for i in n]
        shift_vectors = np.array(list(itertools.product(*nrange3)), dtype=int)
        arr_p1 = p1_wrapped[None, :, :] + (shift_vectors @ rcell)[:, None, :]
        arr_p1 = arr_p1.reshape(-1, 3)
        source, target, dist = _sklearn_query_ball_point(
            p1=arr_p1,
            p2=p2_wrapped,
            max_distance=max_distance,
            return_distance=bool(return_distance),
        )
        image_idx, source = np.divmod(source, l1)
        # The distance vector in the reduced cell is
        #     D = p2[target] - p1[source] + shift_rcell @ rcell,
        # and it is converted to the original cell by `rcell = op @ cell`,
        # i.e. shift = shift_rcell @ op, so that
        #     D = p2[target] - p1[source] + shift @ cell
        shift_rcell = (
            offset2[target] - offset1[source] - shift_vectors[image_idx]
        )
        shift = shift_rcell @ op
        return (
            source.astype(int),
            target.astype(int),
            shift.astype(int),
            dist,
        )


#############################################################
# pymatgen for `distance_pairs`
def _pymatgen_get_distance_pairs(
    p1: np.ndarray,
    p2: np.ndarray | None = None,
    cell: np.ndarray | Cell | None = None,
    max_distance: float = float("inf"),
    return_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(cell, Cell):
        cell = Cell.new(cell=cell)
    p1 = np.asarray(p1, dtype=float)
    if p2 is None:
        p2 = p1
    else:
        p2 = np.asarray(p2, dtype=float)
    assert p1.ndim == p2.ndim == 2
    assert p1.shape[1] == p2.shape[1] == 3
    l1, l2 = p1.shape[0], p2.shape[0]
    max_distance = float(max_distance)
    assert max_distance > 0

    if cell.rank == 0:
        return _sklearn_get_distance_pairs(
            p1=p1,
            p2=p2,
            cell=cell,
            max_distance=max_distance,
            return_distance=return_distance,
        )
    else:
        try:
            import pymatgen  # type: ignore  # noqa: F401
        except ImportError:
            raise ImportError(
                "Cannot import `pymatgen` module. Please install it first."
            )
        try:
            from pymatgen.optimization.neighbors import (  # type: ignore
                find_points_in_spheres,
            )
        except ImportError:
            raise ImportError(
                "Cannot import `find_points_in_spheres` from "
                "`pymatgen.optimization.neighbors` module. Please "
                "upgrade your pymatgen to 2022.0.10 or later."
            )
        target, source, shift, distance = find_points_in_spheres(
            np.ascontiguousarray(p1, dtype=float),
            np.ascontiguousarray(p2, dtype=float),
            r=max_distance,
            pbc=np.ascontiguousarray(cell.array.any(1), dtype=np.int64),
            lattice=np.ascontiguousarray(cell.array, dtype=float),
            tol=1e-8,
        )
        shift = np.asarray(shift, int)
        target = np.asarray(target, int)
        source = np.asarray(source, int)
        distance = np.asarray(distance, float)
        assert distance.shape == (len(source),), "Impossible"
        assert source.shape == (len(source),), "Impossible"
        assert target.shape == (len(target),), "Impossible"
        assert shift.shape == (len(source), 3), "Impossible"
        assert source.max() < l1 and target.max() < l2
        self_pair = (source == target) & (distance <= 1e-5)
        cond = np.logical_not(self_pair)  # exclude self pair
        return (
            source[cond],
            target[cond],
            shift[cond] if cell.rank != 0 else None,  # type: ignore
            distance[cond] if return_distance else None,
        )


#############################################################
# interface for `distance_pairs`
def distance_pairs(
    quantities: str,
    p1: np.ndarray,
    p2: np.ndarray | None = None,
    cell: np.ndarray | Cell | None = None,
    cutoff: float = 6.0,
    backend: str | Literal["sklearn", "pmg", "pymatgen"] = "pmg",
) -> tuple[np.ndarray, ...]:
    """Compute distance pairs for two points set.

    Parameters
    ----------
    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are:

        * 'i' : first atom index
        * 'j' : second atom index
        * 'd' : absolute distance
        * 'D' : distance vector
        * 'S' : shift vector (number of cell boundaries crossed by the bond
            between atom i and j). With the shift vector S, the
            distances D between atoms can be computed from:
                D = p2[j] - p1[i] + S @ a.cell
    p1: np.ndarray
        the first points set.
    p2: np.ndarray | None
        the second points set. if none, p2 is p1.
    cell: np.ndarray | Cell | None
        the periodic lattice.
    cutoff: float
        This is a global cutoff for all points pair.
    backend: str
        The backend for computation.

    Returns:
    -------
    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.
    """
    p1 = np.asarray(p1, dtype=float)
    if p2 is None:
        p2 = p1
    else:
        p2 = np.asarray(p2, dtype=float)
    assert p1.ndim == p2.ndim == 2
    assert p1.shape[1] == p2.shape[1] == 3
    cell = Cell.new(cell)
    if cell.rank != 0:
        assert np.isfinite(cutoff), (
            "The `cutoff` must is finite number for periodic cell."
        )

    if backend.lower() == "sklearn":
        i, j, S, d = _sklearn_get_distance_pairs(
            p1=p1,
            p2=p2,
            cell=cell,
            max_distance=cutoff,
            return_distance=bool("d" in quantities),
        )
    elif backend.lower() in ("pmg", "pymatgen"):
        j, i, S, d = _pymatgen_get_distance_pairs(
            p1=p2,
            p2=p1,
            cell=cell,
            max_distance=cutoff,
            return_distance=bool("d" in quantities),
        )
        # Swap i-j & p1-p2 for that the result match:
        #       D = p2[j] - p1[i] + S @ a.cell
    else:
        raise NotImplementedError(
            f"Invalid backend='{backend}', only 'sklearn',"
            " 'pmg' and 'pymatgen' are supported."
        )
    dct = dict(zip("ijS", [i, j, S]))
    if "D" in quantities:
        dct["D"] = p2[j] - p1[i] + S @ cell
    if "d" in quantities:
        assert isinstance(d, np.ndarray)
        dct["d"] = d  # absolute distance
    return tuple(dct[q] for q in quantities)


#############################################################
# ASE for `neighbor_list`
def _ase_neighbor_list_kdtree(
    quantities: str,
    a: Atoms,
    cutoff: float,
    indices: np.ndarray | None = None,
    self_interaction: bool = False,
    bothways: bool = True,
    sorted: bool = True,
) -> tuple[np.ndarray, ...]:
    if indices is None:
        indices = np.arange(len(a))
    nl = __build_neighbor_list_ase(
        atoms=a,
        cutoffs=[float(cutoff / 2)] * len(a),
        self_interaction=self_interaction,
        bothways=bothways,
        sorted=sorted,
        skin=0,
    )
    i = np.array([], dtype=int)
    j = np.array([], dtype=int)
    S = np.zeros(shape=(0, 3), dtype=int)
    for idx in indices:
        jj, SS = nl.get_neighbors(idx)
        j = np.append(j, jj)
        S = np.vstack([S, SS])
        i = np.append(i, np.full(len(jj), idx))
    dct: dict[str, np.ndarray] = {
        "i": i,
        "j": j,
        "S": S,
    }
    if "d" in quantities.lower():
        v = a.positions[j] - a.positions[i]
        dct["D"] = D = v + S @ a.cell
        if "d" in quantities:
            dct["d"] = np.linalg.norm(D, axis=1)
    return tuple(dct[q] for q in quantities)


#############################################################
# matscipy for `neighbor_list`
def _matscipy_neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
) -> tuple[np.ndarray, ...]:
    try:
        from matscipy.neighbours import (  # type: ignore  # noqa: F401
            neighbour_list as _matscipy_neighbour_list,
        )
    except Exception:
        raise Exception(
            "Please install `matscipy` to speed up "  #
            "the calculation of neighbor list.",
        )
    if a.cell.rank != 0:
        return tuple(
            _matscipy_neighbour_list(
                quantities=quantities,
                cutoff=cutoff,
                atoms=a,
            )
        )
    else:
        return distance_pairs(
            quantities,
            p1=a.positions,
            p2=a.positions,
            cutoff=float(cutoff),
            backend="sklearn",
            cell=None,
        )


#############################################################
# VESIN for `neighbor_list`
def _vesin_neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
    self_interaction: bool = False,
    max_nbins: int = 0,
) -> tuple[np.ndarray, ...]:
    try:
        from vesin import ase_neighbor_list  # type: ignore
    except Exception:
        raise Exception(
            "Please install `vesin` to speed up "  #
            "the calculation of neighbor list.",
        )
    return tuple(
        ase_neighbor_list(
            quantities=quantities,
            a=a,
            cutoff=cutoff,
            self_interaction=self_interaction,
            max_nbins=max_nbins,
        )
    )


#############################################################
# interface for `neighbor_list`
def neighbor_list(
    quantities: str,
    a: Atoms,
    cutoff: float,
    self_interaction: bool = False,
    max_nbins: int = int(1e6),
    bothways: bool = True,
    sorted: bool = True,
    indices: ArrayLike | None = None,
    backend: str
    | Literal[
        "vesin",
        "matscipy",
        "ase_3loop",
        "ase_kdtree",
        "pymatgen",
        "sklearn",
    ] = "sklearn",
) -> tuple[np.ndarray, ...]:
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters
    ----------
    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are:

        * 'i' : first atom index
        * 'j' : second atom index
        * 'd' : absolute distance
        * 'D' : distance vector
        * 'S' : shift vector (number of cell boundaries crossed by the bond
            between atom i and j). With the shift vector S, the
            distances D between atoms can be computed from:
            D = a.positions[j] - a.positions[i] + S @ a.cell
    a: :class:`ase.Atoms`
        Atomic configuration.
    cutoff: float or dict
        This is a global cutoff for all elements.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.
    sorted: bool
        Sort the neighbor list by first atom index 'i', second atom
        index 'j' and shift vector 'S'. Default: True
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.
    indices: np.ndarray | None
        Indices of atoms to use in the neighbor list.
        If None, all atoms are.
    backend: str
        The backend for computation.

    Returns:
    -------
    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.
    """
    assert isinstance(a, Atoms), (
        f"The `a` must be ase.Atoms, but"  #
        f" {a.__class__.__name__} got"
    )
    assert all(i in "ijdDS" for i in quantities), (
        "The `quantities` must be an elements of "  #
        f"'ijdDS', but got {quantities},"
    )
    old_quantities = quantities
    for s in "ijS":
        if s not in quantities:
            quantities += s

    if len(a) <= 1 and a.cell.rank == 0:
        return tuple(np.array([]) for _ in quantities)

    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        indices = np.unique(indices.flatten())
        if len(indices) == 0:
            indices = None
        else:
            assert 0 <= indices.min() <= indices.max() < len(a), (
                "The indices must be in range [0, len(a))."
            )
            if len(indices) == len(a):
                indices = None
    if indices is not None:
        assert bothways is True, (
            "Only support `bothways=True` "  #
            "for `indices` not None."
        )

    msg = "The {} is not support for non-empty indices."
    if backend == "ase_kdtree":
        return _ase_neighbor_list_kdtree(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            indices=indices,
            self_interaction=self_interaction,
            bothways=bothways,
            sorted=sorted,
        )
    elif backend == "ase_3loop":
        assert indices is None, msg.format(backend)
        result = _ase_neighbor_list_3loop(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            self_interaction=self_interaction,
            max_nbins=int(max_nbins),
        )  # type: ignore
    elif backend in ("sklearn", "pmg", "pymatgen"):
        if indices is None:
            indices = np.arange(len(a))
        result = distance_pairs(
            quantities=quantities,
            p1=a.positions[indices],
            p2=a.positions,
            cell=a.cell,
            cutoff=cutoff,
            backend=backend,
        )
    elif backend == "vesin":
        assert indices is None, msg.format(backend)
        result = _vesin_neighbor_list(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
            self_interaction=self_interaction,
            max_nbins=int(max_nbins),
        )  # type: ignore
    elif backend == "matscipy":
        result = _matscipy_neighbor_list(
            quantities=quantities,
            a=a,
            cutoff=float(cutoff),
        )
    else:
        raise RuntimeError(f"Unknown neighbor list backend: {backend}")

    dct = {k: result[i] for i, k in enumerate(quantities)}
    assert set(dct.keys()) >= set("ijS"), "Impossible."
    i, j, (Sx, Sy, Sz) = dct["i"], dct["j"], dct["S"].T
    cond = np.ones(len(i), dtype=bool)
    if not self_interaction:
        self_interaction_cond = (Sx == 0) & (Sy == 0) & (Sz == 0)
        self_interaction_cond = self_interaction_cond & (i == j)
        cond[self_interaction_cond] = False
    if not bothways:
        bothways_cond_1 = Sx < 0
        bothways_cond_2 = (Sx == 0) & (Sy < 0)
        bothways_cond_3 = (Sx == 0) & (Sy == 0) & (Sz < 0)
        bothways_cond_4 = (Sx == 0) & (Sy == 0) & (Sz == 0) & (i > j)
        cond[bothways_cond_4] = False
        cond[bothways_cond_3] = False
        cond[bothways_cond_2] = False
        cond[bothways_cond_1] = False
    dct = {k: v[cond] for k, v in dct.items()}
    if sorted:
        i, j, (Sx, Sy, Sz) = dct["i"], dct["j"], dct["S"].T
        idx: np.ndarray = np.lexsort((i, j, Sz, Sy, Sx))
        dct = {k: v[idx] for k, v in dct.items()}
    return tuple(dct[k] for k in old_quantities)
