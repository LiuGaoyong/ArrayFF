"""Per-element parameters for pairwise potentials."""

from __future__ import annotations

import dataclasses as dc
from typing import Any

import numpy as np

from arrayff.utils.dispatch import to_backend

__all__ = ["PairParameters"]


@dc.dataclass(frozen=True)
class PairParameters:
    """Element ordered parameters plus mixing rules.

    Attributes
    ----------
    elements:
        Sorted unique atomic numbers, ``(S,)``.
    values:
        ``{name: (S,) array}`` with the per element parameters.
    mixing:
        ``{name: rule}`` overriding the form's defaults.
    cutoff:
        Optional ``(S,)`` per element cutoff radius, mixed pairwise just like
        the other parameters (this is what the OpenKIM universal model wants).
    """

    elements: np.ndarray
    values: dict[str, Any]
    mixing: dict[str, str]
    cutoff: Any | None = None

    def to_backend(self, backend: str) -> PairParameters:
        if backend == "numpy":
            return self
        return PairParameters(
            elements=self.elements,
            values={
                name: to_backend(value, backend)
                for name, value in self.values.items()
            },
            mixing=self.mixing,
            cutoff=None
            if self.cutoff is None
            else to_backend(self.cutoff, backend),
        )
