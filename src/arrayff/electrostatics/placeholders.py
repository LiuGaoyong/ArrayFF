"""Declared but not yet implemented charge models."""

from __future__ import annotations

from arrayff.utils.exceptions import ScopeError

__all__ = ["EEM", "PQEq"]


class _Placeholder:
    def __init__(self, *args, **kwargs) -> None:
        raise ScopeError(
            f"{type(self).__name__} is declared in the API but not implemented "
            "yet; only QEq is available."
        )


class EEM(_Placeholder):
    """Electronegativity equalization method (planned)."""


class PQEq(_Placeholder):
    """Polarizable charge equilibration with core/shell (planned)."""
