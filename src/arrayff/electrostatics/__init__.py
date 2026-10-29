"""Electrostatic charge models (QEq implemented; EEM/PQEq planned)."""

from __future__ import annotations

import importlib
from typing import Any

from arrayff.electrostatics.placeholders import EEM, PQEq
from arrayff.electrostatics.qeq import QEq

__all__ = ["EEM", "PQEq", "QEq", "QEqCalculator"]


def __getattr__(name: str) -> Any:
    if name == "QEqCalculator":
        module = importlib.import_module("arrayff.electrostatics.calculator")
        value = module.QEqCalculator
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
