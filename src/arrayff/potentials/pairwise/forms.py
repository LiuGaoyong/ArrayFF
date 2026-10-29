"""Pairwise functional forms.

A :class:`PairForm` only knows how to turn a distance (and already mixed
per-element parameters) into an energy. Cutoffs, shifts, smoothing, table
handling and forces are handled once by
:class:`arrayff.potentials.pairwise.base.PairwisePotential`.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

from arrayff.utils.dispatch import do

__all__ = [
    "MIXERS",
    "GeneralizedLJ",
    "Morse",
    "PairForm",
]


def _arithmetic(a: Any, b: Any) -> Any:
    return 0.5 * (a + b)


def _geometric(a: Any, b: Any) -> Any:
    return do("sqrt", a * b)


#: Mixing rules usable for any parameter name.
MIXERS: dict[str, Callable[[Any, Any], Any]] = {
    "arithmetic": _arithmetic,
    "geometric": _geometric,
    "none": lambda a, b: a,
}


class PairForm(abc.ABC):
    """Functional form of a pair interaction."""

    name: str = "pair"
    param_names: tuple[str, ...] = ()
    default_mixing: dict[str, str] = {}

    def mix(
        self,
        first: dict[str, Any],
        second: dict[str, Any],
        rules: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Combine per-species parameters into per-pair parameters."""
        rules = rules or {}
        mixed: dict[str, Any] = {}
        for name in self.param_names:
            rule = rules.get(name, self.default_mixing.get(name, "arithmetic"))
            try:
                mixer = MIXERS[rule]
            except KeyError:
                raise KeyError(
                    f"Unknown mixing rule {rule!r} for parameter {name!r}"
                ) from None
            mixed[name] = mixer(first[name], second[name])
        return mixed

    @abc.abstractmethod
    def energy(self, r: Any, params: dict[str, Any]) -> Any:
        """Return the uncut pair energy for distances ``r``."""

    def cutoff_energy(self, rc: Any, params: dict[str, Any]) -> Any:
        """Energy at the cutoff, used for the shifted convention."""
        return self.energy(rc, params)


class GeneralizedLJ(PairForm):
    r"""Generalized Lennard-Jones ``(m, n)`` form.

    .. math::

        V(r) = \varepsilon \left[
            \frac{n}{m-n}\left(\frac{r_0}{r}\right)^m
            - \frac{m}{m-n}\left(\frac{r_0}{r}\right)^n
        \right]

    with the minimum at ``r0`` and well depth ``depth``. ``(m, n) = (12, 6)``
    is the usual Lennard-Jones potential and ``(9, 3)`` the surface potential.
    """

    param_names = ("r0", "depth")
    default_mixing = {"r0": "arithmetic", "depth": "geometric"}

    def __init__(self, m: int = 12, n: int = 6) -> None:
        if not 0 < n < m:
            raise ValueError("GeneralizedLJ requires 0 < n < m")
        self.m = int(m)
        self.n = int(n)
        self.name = f"lj{m}-{n}"

    def energy(self, r: Any, params: dict[str, Any]) -> Any:
        x = params["r0"] / r
        prefactor = params["depth"]
        return prefactor * (
            (self.n / (self.m - self.n)) * x**self.m
            - (self.m / (self.m - self.n)) * x**self.n
        )


class Morse(PairForm):
    """Morse potential ``depth * (e^{-2a(r-r0)} - 2 e^{-a(r-r0)})``."""

    name = "morse"
    param_names = ("r0", "depth", "a")
    default_mixing = {
        "r0": "arithmetic",
        "depth": "geometric",
        "a": "arithmetic",
    }

    def energy(self, r: Any, params: dict[str, Any]) -> Any:
        exponent = do("exp", -params["a"] * (r - params["r0"]))
        return params["depth"] * (exponent * exponent - 2.0 * exponent)
