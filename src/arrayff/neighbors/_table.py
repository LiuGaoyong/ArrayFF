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
from arrayff.utils.dispatch import to_backend

__all__ = ["NeighborPaddedTable", "NeighborEdgeList"]


@dc.dataclass(frozen=True)
class NeighborListABC(ABC):
    """Base class for neighbour tables."""

    def astype(self, dtype: DTypeLike = int) -> Self:
        """Return a copy whose arrays have the specified dtype."""
        return self.__class__(
            **{
                k: np.asarray(getattr(self, k), dtype=dtype)
                for k in self.__dataclass_fields__.keys()
            }
        )

    def to_backend(self, backend: str, dtype: DTypeLike = int) -> Self:
        """Return a copy whose arrays live on `backend`."""
        obj = self.astype(dtype=dtype)
        if backend == "numpy":
            return obj
        else:
            return self.__class__(
                **{
                    k: to_backend(getattr(obj, k), backend)
                    for k in self.__dataclass_fields__.keys()
                }
            )

    @abstractmethod
    @classmethod
    def empty(cls, *args, **kwargs) -> Self:
        pass

    @abstractmethod
    @classmethod
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
        """Return the combined shift array."""
        lst = [self.shift_x, self.shift_y, self.shift_z]
        return np.column_stack(lst).astype(int)

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
            return cls.empty()
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
        return int(np.sum(self.mask))

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
            return cls.empty(n_atoms)
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
            return cls.empty(n_atoms)

        counts = np.bincount(i, minlength=n_atoms)
        width = int(np.max(counts, initial=0))
        if max_neighbors is not None:
            if int(max_neighbors) < width:
                raise ValueError(
                    f"max_neighbors={max_neighbors} is smaller than the largest "
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
