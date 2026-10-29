"""Generic ASE calculator adapter.

Every potential can be driven through :class:`ase.Calculator`, so
``Atoms.get_potential_energy()`` / ``get_forces()`` (and MD, optimizers,
NEB, ...) work unchanged.  Charge models additionally expose
``results["charges"]``.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

__all__ = ["ArrayFFCalculator", "split_kwargs"]

#: Keywords consumed by :class:`ase.calculators.calculator.Calculator`.
_CALCULATOR_PARAMETERS = {
    "label",
    "directory",
    "restart",
    "discard",
    "logfile",
    "use_cache",
    "txt",
    "communicator",
}


def split_kwargs(
    kwargs: dict[str, Any],
    potential_cls: type,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``kwargs`` into potential arguments and ASE calculator options.

    Potential classes have explicit signatures; anything else that ASE's
    ``Calculator`` recognises (or unknown keys) is forwarded to it, so new
    potential options such as ``jit``/``method`` are not silently dropped.
    """
    parameters = inspect.signature(potential_cls.__init__).parameters
    potential_kwargs: dict[str, Any] = {}
    calculator_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in parameters and key not in _CALCULATOR_PARAMETERS:
            potential_kwargs[key] = value
        else:
            calculator_kwargs[key] = value
    return potential_kwargs, calculator_kwargs


class ArrayFFCalculator(Calculator):
    """Wrap any arrayff potential as an ASE calculator."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        potential: Any,
        *,
        method: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.potential = potential
        self.method = method
        supported = ["energy", "forces", "stress"]
        if getattr(potential, "supports_charges", False):
            supported.append("charges")
        # ASE reads this attribute per instance.
        self.implemented_properties = supported

    def get_stress(self, atoms: Any = None) -> Any:
        """Return a zero Voigt stress vector for non-periodic systems.

        ASE always exposes an ``Atoms.cell`` object, even for molecules.
        Asking for stress there is not meaningful; return zeros instead of
        raising, and do not add a ``stress`` entry to ``results``.
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None or atoms.cell.rank != 3:
            # Non-periodic stress is undefined; return a zero Voigt vector
            # without adding a ``stress`` entry to ``results`` (reference
            # data stores no periodic stress for these systems).
            return np.zeros(6)
        return super().get_stress(atoms)

    def calculate(
        self,
        atoms: Any = None,
        properties: Iterable[str] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        requested = set(properties)
        if (
            "stress" in requested
            and self.atoms is not None
            and self.atoms.cell is not None
            and self.atoms.cell.rank != 3
        ):
            requested.discard("stress")
        wanted = ["energy"]
        if "forces" in requested:
            wanted.append("forces")
        if "stress" in requested:
            wanted.append("stress")
        if getattr(self.potential, "supports_charges", False):
            # Preserve the historical behaviour: a charge model always
            # exposes its charges after a calculation, even for an
            # energy-only request.
            wanted.append("charges")

        if hasattr(self.potential, "evaluate"):
            result = self.potential.evaluate(
                self.atoms, properties=tuple(wanted), method=self.method
            )
            energy = result["energy"]
            forces = result.get("forces")
            stress = result.get("stress")
            charges = result.get("charges")
        else:  # backward compatibility for non-PotentialBase objects
            energy, forces = self.potential.energy_and_forces(self.atoms)
            stress = (
                self.potential.stress(self.atoms)
                if "stress" in requested
                else None
            )
            charges = (
                self.potential.charges(self.atoms)
                if "charges" in requested
                else None
            )

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        if forces is not None:
            self.results["forces"] = forces
        if stress is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)
        if charges is not None:
            self.results["charges"] = charges
