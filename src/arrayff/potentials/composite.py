"""Additive composite potentials.

``LJ(...) + QEq(...)`` returns a :class:`CompositePotential` whose energy,
forces and stress are the sum of its components.  Each component keeps its
own neighbour table, backend and evaluation method; a composite is never
compiled as a whole (set ``jit=True`` on the components instead).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from arrayff.potentials.base import PotentialBase
from arrayff.utils.exceptions import ScopeError

__all__ = ["CompositePotential"]


class CompositePotential(PotentialBase):
    """Sum of two or more potentials / charge models."""

    def __init__(
        self,
        components: tuple[PotentialBase, ...] | list[PotentialBase],
    ) -> None:
        flattened: list[PotentialBase] = []
        for component in components:
            if isinstance(component, CompositePotential):
                flattened.extend(component.components)
            else:
                flattened.append(component)
        if not flattened:
            raise ValueError(
                "CompositePotential requires at least one component"
            )
        for component in flattened:
            if not isinstance(component, PotentialBase):
                raise TypeError(
                    "CompositePotential components must be PotentialBase "
                    f"instances, got {type(component).__name__}"
                )
        backends = {component.backend for component in flattened}
        if len(backends) != 1:
            raise TypeError(
                "All composite components must use the same backend, got "
                f"{sorted(backends)}"
            )
        dtypes = {
            component.dtype
            for component in flattened
            if component.dtype is not None
        }
        if len(dtypes) > 1:
            raise TypeError(
                "All composite components must use the same dtype, got "
                f"{dtypes}"
            )
        charge_components = [
            component
            for component in flattened
            if getattr(component, "supports_charges", False)
        ]
        if len(charge_components) > 1:
            raise ValueError(
                "A composite may contain at most one charge model "
                f"({[type(c).__name__ for c in charge_components]})"
            )
        super().__init__(
            backend=next(iter(backends)),
            dtype=next(iter(dtypes), None),
            jit=False,
            method="analytic",
        )
        self.components = tuple(flattened)
        self.supports_charges = bool(charge_components)
        self._charge_component = (
            charge_components[0] if charge_components else None
        )
        self.name = "+".join(
            getattr(component, "name", type(component).__name__)
            for component in self.components
        )

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __repr__(self) -> str:
        return f"CompositePotential({self.name})"

    def evaluate(
        self,
        atoms: Atoms,
        *,
        properties: tuple[str, ...] = ("energy",),
        method: str | None = None,
    ) -> dict[str, Any]:
        props = tuple(properties)
        unknown = set(props) - {"energy", "forces", "stress", "charges"}
        if unknown:
            raise ScopeError(
                f"Unsupported properties for composite: {sorted(unknown)}"
            )
        component_props = tuple(
            dict.fromkeys(("energy", *(p for p in props if p != "charges")))
        )
        results = [
            component.evaluate(
                atoms, properties=component_props, method=method
            )
            for component in self.components
        ]
        out: dict[str, Any] = {
            "energy": float(sum(result["energy"] for result in results))
        }
        if "forces" in props:
            out["forces"] = np.sum(
                [result["forces"] for result in results], axis=0
            )
        if "stress" in props:
            out["stress"] = np.sum(
                [result["stress"] for result in results], axis=0
            )
        if "charges" in props:
            if self._charge_component is None:
                raise ScopeError("This composite has no charge model")
            out["charges"] = self._charge_component.charges(atoms)
        return out

    def charges(self, atoms: Atoms) -> np.ndarray:
        if self._charge_component is None:
            raise ScopeError("This composite has no charge model")
        return self._charge_component.charges(atoms)

    def clear_cache(self) -> None:
        for component in self.components:
            component.clear_cache()
