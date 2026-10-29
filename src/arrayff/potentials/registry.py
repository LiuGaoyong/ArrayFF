"""Potential registration.

The registry drives the public ``arrayff.get_potential`` helper and records
which potentials are implemented, so placeholder modules can be shipped
without pretending to work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["available_potentials", "get_potential", "register_potential"]

_REGISTRY: dict[str, tuple[Callable[..., Any], bool]] = {}


def register_potential(
    name: str, *, implemented: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a potential class under ``name``."""

    def decorator(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise KeyError(f"Potential {name!r} is already registered")
        _REGISTRY[name] = (cls, implemented)
        return cls

    return decorator


def get_potential(name: str, **kwargs: Any) -> Any:
    """Instantiate the potential registered under ``name``."""
    try:
        cls, implemented = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown potential {name!r}; available: {sorted(_REGISTRY)}"
        ) from None
    if not implemented:
        raise NotImplementedError(
            f"Potential {name!r} is declared but not implemented yet."
        )
    return cls(**kwargs)


def available_potentials() -> dict[str, bool]:
    """Return ``{name: implemented}`` for every known potential."""
    return {name: flag for name, (_, flag) in sorted(_REGISTRY.items())}
