"""ASE calculator interface.

Every potential in arrayff can be driven through :class:`ase.Calculator`, so
``Atoms.get_potential_energy()`` / ``get_forces()`` (and MD, optimizers,
NEB, ...) work unchanged. Charge models additionally expose ``results
["charges"]``.
"""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator, all_changes

__all__ = [
    "ArrayFFCalculator",
    "EAMCalculator",
    "EMTCalculator",
    "LJCalculator",
    "QEqCalculator",
]


class ArrayFFCalculator(Calculator):
    """Wrap any arrayff potential as an ASE calculator."""

    implemented_properties = ["energy", "forces", "stress", "charges"]

    def __init__(self, potential: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.potential = potential

    def calculate(
        self,
        atoms: Any = None,
        properties: tuple[str, ...] | list[str] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        energy, forces = self.potential.energy_and_forces(self.atoms)
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        if "stress" in properties and hasattr(self.potential, "stress"):
            from ase.stress import full_3x3_to_voigt_6_stress

            self.results["stress"] = full_3x3_to_voigt_6_stress(
                self.potential.stress(self.atoms)
            )
        if hasattr(self.potential, "charges"):
            self.results["charges"] = self.potential.charges(self.atoms)


class LJCalculator(ArrayFFCalculator):
    """Lennard-Jones calculator (OpenKIM universal parameters by default)."""

    def __init__(self, **kwargs: Any) -> None:
        from arrayff.potentials import LennardJones

        potential_keys = {
            "sigma",
            "epsilon",
            "cutoff",
            "shift",
            "backend",
            "dtype",
            "max_neighbors",
        }
        potential_args = {
            key: value for key, value in kwargs.items() if key in potential_keys
        }
        super().__init__(LennardJones(**potential_args))
        self.parameters = potential_args


class EMTCalculator(ArrayFFCalculator):
    """Effective medium theory calculator."""

    def __init__(self, **kwargs: Any) -> None:
        from arrayff.potentials import EMT

        super().__init__(EMT(**kwargs))


class EAMCalculator(ArrayFFCalculator):
    """EAM calculator reading a ``setfl`` file."""

    def __init__(self, potential: str, **kwargs: Any) -> None:
        from arrayff.potentials import EAM

        super().__init__(EAM(potential, **kwargs))


class QEqCalculator(ArrayFFCalculator):
    """Charge equilibration calculator (exposes ``charges``)."""

    def __init__(self, **kwargs: Any) -> None:
        from arrayff.electrostatics import QEq

        super().__init__(QEq(**kwargs))
