"""Neighbour tables in a backend friendly, statically shaped form.

Kernels need fixed shape neighbour arrays so that JAX can trace them, while
:mod:`arrayff.neighborlist` (sklearn backend by default) returns the usual
flat ``(i, j, S)`` edge list. This module converts one into the other with
vectorised numpy operations only: no Python loop over atoms or pairs.
"""

from __future__ import annotations

import dataclasses as dc
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, Self, override

import numpy as np
from ase import Atoms
from numpy.typing import DTypeLike

from arrayff.neighbors._list import NeighborListBackendType, neighbor_list
from arrayff.utils.backend import infer_backend
from arrayff.utils.dispatch import do, to_backend, to_numpy

__all__ = ["NeighborPaddedTable", "NeighborEdgeList"]


@dc.dataclass(frozen=True)
class NeighborListABC(ABC):
    """Base class for neighbour tables."""

    def _first_array(self) -> np.ndarray:
        """Return the first stored array (used for backend detection)."""
        return getattr(self, next(iter(self.__dataclass_fields__)))

    def astype(self, dtype: DTypeLike = int) -> Self:
        """Return a copy whose arrays have the specified dtype.

        Only defined for numpy backed tables; convert to a backend *after*
        fixing the dtype (:meth:`to_backend` does this for you).
        """
        if infer_backend(self._first_array()) != "numpy":
            raise TypeError(
                "astype() is only defined for numpy backed tables; "
                "call it before to_backend()."
            )
        values: dict[str, np.ndarray] = {}
        for name in self.__dataclass_fields__:
            array = np.asarray(getattr(self, name))
            values[name] = array.astype(bool if name == "mask" else dtype)
        return self.__class__(**values)

    def to_backend(self, backend: str, dtype: DTypeLike = int) -> Self:
        """Return a copy whose arrays live on `backend`."""
        current = infer_backend(self._first_array())
        if current == backend:
            return self
        if current != "numpy":
            # Detach from the current backend first (torch/jax -> numpy).
            detached = self.__class__(
                **{
                    name: to_numpy(getattr(self, name))
                    for name in self.__dataclass_fields__
                }
            )
            if backend == "numpy":
                return detached
            return detached.to_backend(backend, dtype=dtype)
        obj = self.astype(dtype=dtype)
        if backend == "numpy":
            return obj
        return self.__class__(
            **{
                name: to_backend(getattr(obj, name), backend)
                for name in self.__dataclass_fields__
            }
        )

    @classmethod
    @abstractmethod
    def empty(cls, *args, **kwargs) -> Self:
        pass

    @classmethod
    @abstractmethod
    def from_ase(
        cls,
        atoms: Atoms,
        cutoff: float,
        *,
        backend: str | Literal["numpy", "jax", "torch"] = "numpy",
        builder: NeighborListBackendType = "sklearn",
        **kwargs,
    ) -> Self:
        pass


@dc.dataclass(frozen=True)
class NeighborEdgeList(NeighborListABC):
    source: np.ndarray
    target: np.ndarray
    shift_x: np.ndarray
    shift_y: np.ndarray
    shift_z: np.ndarray

    @cached_property
    def shift(self) -> np.ndarray:
        """Return the combined ``(E, 3)`` shift array.

        Stays on the table backend: ``np.column_stack`` would silently detach
        (or fail on) torch/jax tensors.
        """
        parts = [self.shift_x, self.shift_y, self.shift_z]
        if infer_backend(self.shift_x) == "numpy":
            return np.column_stack(parts).astype(np.int64)
        return do("stack", parts, axis=-1)

    def __post_init__(self) -> None:
        msg = r"The shape of {} must be {}, but got {}."
        for v, shape, name in [
            (self.source, (len(self),), "source"),
            (self.target, (len(self),), "target"),
            (self.shift_x, (len(self),), "shift_x"),
            (self.shift_y, (len(self),), "shift_y"),
            (self.shift_z, (len(self),), "shift_z"),
        ]:
            assert v.shape == shape, msg.format(name, shape, v.shape)

    def __len__(self) -> int:
        return int(self.source.shape[0])

    @classmethod
    @override
    def empty(cls) -> Self:
        return cls(
            source=np.empty((0,), dtype=int),
            target=np.empty((0,), dtype=int),
            shift_x=np.empty((0,), dtype=int),
            shift_y=np.empty((0,), dtype=int),
            shift_z=np.empty((0,), dtype=int),
        )

    @classmethod
    @override
    def from_ase(
        cls,
        atoms: Atoms,
        cutoff: float,
        *,
        backend: str | Literal["numpy", "jax", "torch"] = "numpy",
        builder: NeighborListBackendType = "sklearn",
        **kwargs,
    ) -> Self:
        """Build a padded neighbour table for ``atoms``.

        Parameters
        ----------
        cutoff:
            Global cutoff radius in Angstrom.
        backend:
            Backend to use for the table. Defaults to ``"numpy"``.
        builder:
            Neighbour list implementation, ``"sklearn"`` (default) or any other
            backend supported by :func:`arrayff.neighborlist.neighbor_list`.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(
                f"Expected an ase.Atoms instance,"  #
                f" but got {type(atoms).__name__}."
            )
        n_atoms = len(atoms)
        if n_atoms == 0 or cutoff <= 0.0:
            return cls.empty().to_backend(backend)
        i, j, shift = neighbor_list(
            "ijS",
            a=atoms,
            backend=builder,
            cutoff=float(cutoff),
            self_interaction=False,
            bothways=True,
            sorted=False,
        )
        x, y, z = shift.T
        return (
            cls(
                source=i,
                target=j,
                shift_x=x,
                shift_y=y,
                shift_z=z,
            )
            .astype(int)
            .to_backend(backend)
        )

    def to_padded(
        self,
        n_atoms: int,
        *,
        max_neighbors: int | None = None,
    ) -> NeighborPaddedTable:
        """Return a padded view of this edge list.

        The conversion is a pure table operation (integer indices only);
        values are copied to the current backend afterwards.
        """
        backend = infer_backend(self.source)
        source = np.asarray(to_numpy(self.source), dtype=np.int64)
        target = np.asarray(to_numpy(self.target), dtype=np.int64)
        shift = np.asarray(to_numpy(self.shift), dtype=np.int64)
        if len(source) == 0:
            return NeighborPaddedTable.empty(n_atoms).to_backend(backend)

        counts = np.bincount(source, minlength=n_atoms)
        width = int(np.max(counts, initial=0))
        if max_neighbors is not None:
            if int(max_neighbors) < width:
                raise ValueError(
                    f"max_neighbors={max_neighbors} is smaller than the "
                    f"largest neighbour count ({width})."
                )
            width = int(max_neighbors)

        order = np.argsort(source, kind="stable")
        rows = source[order]
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
        cols = np.arange(len(order), dtype=np.int64) - offsets[rows]

        index = np.zeros((n_atoms, width), dtype=np.int64)
        table_shift = np.zeros((n_atoms, width, 3), dtype=np.int64)
        mask = np.zeros((n_atoms, width), dtype=bool)
        index[rows, cols] = target[order]
        table_shift[rows, cols] = shift[order]
        mask[rows, cols] = True
        table = NeighborPaddedTable(index=index, shift=table_shift, mask=mask)
        return table.to_backend(backend)


@dc.dataclass(frozen=True)
class NeighborPaddedTable(NeighborListABC):
    """Padded neighbour table for each atom.

    Attributes
    ----------
    index:
        ``(N, M)`` int array of neighbour indices; padding entries point at
        atom 0 and must be ignored via ``mask``.
    shift:
        ``(N, M, 3)`` int array of periodic image shifts ``S`` such that
        ``d_ij = r_j - r_i + S @ cell``.
    mask:
        ``(N, M)`` bool array, ``True`` for real neighbours.
    """

    index: np.ndarray
    shift: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        msg = r"The shape of `{}` must be {}, but got {}."
        assert self.index.ndim == 2, "The `index` must be 2D array."
        for v, shape, name in [
            (self.shift, self.index.shape + (3,), "shift"),
            (self.mask, self.index.shape, "mask"),
        ]:
            assert v.shape == shape, msg.format(name, shape, v.shape)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    @property
    def n_atoms(self) -> int:
        return self.__len__()

    @property
    def max_neighbors(self) -> int:
        return int(self.index.shape[1])

    @property
    def n_neighbors_total(self) -> int:
        return int(np.asarray(to_numpy(self.mask)).sum())

    @classmethod
    @override
    def empty(cls, n_atoms: int) -> Self:
        return cls(
            index=np.empty((n_atoms, 0), dtype=int),
            shift=np.empty((n_atoms, 0, 3), dtype=int),
            mask=np.zeros((n_atoms, 0), dtype=bool),
        )

    @classmethod
    @override
    def from_ase(
        cls,
        atoms: Atoms,
        cutoff: float,
        *,
        backend: str | Literal["numpy", "jax", "torch"] = "numpy",
        builder: NeighborListBackendType = "sklearn",
        max_neighbors: int | None = None,
        **kwargs,
    ) -> Self:
        """Build a padded neighbour table for ``atoms``.

        Parameters
        ----------
        cutoff:
            Global cutoff radius in Angstrom.
        backend:
            Backend to use for the table. Defaults to ``"numpy"``.
        builder:
            Neighbour list implementation, ``"sklearn"`` (default) or any other
            backend supported by :func:`arrayff.neighborlist.neighbor_list`.
        max_neighbors:
            Fixed second dimension of the table. Required for JAX when the shape
            must stay constant between calls; defaults to the largest neighbour
            count found.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError(
                f"Expected an ase.Atoms instance,"  #
                f" but got {type(atoms).__name__}."
            )
        n_atoms = len(atoms)
        if n_atoms == 0 or cutoff <= 0.0:
            return cls.empty(n_atoms).to_backend(backend)
        i, j, shift = neighbor_list(
            "ijS",
            a=atoms,
            backend=builder,
            cutoff=float(cutoff),
            self_interaction=False,
            bothways=True,
            sorted=False,
        )
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        shift = np.asarray(shift, dtype=np.int64)
        if len(i) == 0:
            return cls.empty(n_atoms).to_backend(backend)

        counts = np.bincount(i, minlength=n_atoms)
        width = int(np.max(counts, initial=0))
        if max_neighbors is not None:
            if int(max_neighbors) < width:
                raise ValueError(
                    f"max_neighbors={max_neighbors} is smaller than the "
                    f"largest "
                    f"neighbour count ({width})."
                )
            width = int(max_neighbors)

        # Vectorised grouping: sort by centre atom, then compute each entry's
        # column inside its row from the running offsets.
        order = np.argsort(i, kind="stable")
        rows = i[order]
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
        cols = np.arange(len(order), dtype=np.int64) - offsets[rows]

        index = np.zeros((n_atoms, width), dtype=np.int64)
        table_shift = np.zeros((n_atoms, width, 3), dtype=np.int64)
        mask = np.zeros((n_atoms, width), dtype=bool)
        index[rows, cols] = j[order]
        table_shift[rows, cols] = shift[order]
        mask[rows, cols] = True
        table = cls(index=index, shift=table_shift, mask=mask)
        return table.to_backend(backend)

    def to_edgelist(self) -> NeighborEdgeList:
        """Flatten the padded table into an edge list (numpy tables only)."""
        if infer_backend(self.index) != "numpy":
            raise TypeError(
                "to_edgelist() requires a numpy backed table; convert the "
                "result afterwards."
            )
        n_atoms, width = self.index.shape
        mask = np.asarray(self.mask, dtype=bool)
        rows = np.broadcast_to(np.arange(n_atoms)[:, None], (n_atoms, width))
        source = rows[mask].astype(np.int64)
        target = np.asarray(self.index)[mask].astype(np.int64)
        shift = np.asarray(self.shift)[mask].astype(np.int64)
        return NeighborEdgeList(
            source=source,
            target=target,
            shift_x=shift[:, 0],
            shift_y=shift[:, 1],
            shift_z=shift[:, 2],
        )
