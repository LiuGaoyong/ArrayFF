"""Declared but not yet implemented potentials.

They are registered with ``implemented=False`` so the public API advertises the
roadmap honestly: ``get_potential("meam")`` raises ``NotImplementedError`` and
constructing the class directly raises a ``ScopeError``.
"""

from __future__ import annotations

from arrayff.potentials.registry import register_potential
from arrayff.utils.exceptions import ScopeError

__all__ = ["MEAM", "UF3"]


class _Placeholder:
    name = "placeholder"

    def __init__(self, *args, **kwargs) -> None:
        raise ScopeError(
            f"{type(self).__name__} is declared in the API but not implemented "
            "yet; see docs/architecture.md for the roadmap."
        )


@register_potential("meam", implemented=False)
class MEAM(_Placeholder):
    """Second nearest neighbour MEAM (planned)."""

    name = "meam"


@register_potential("uf3", implemented=False)
class UF3(_Placeholder):
    """UF3 spline force field (planned)."""

    name = "uf3"
