"""File readers for tabulated potentials."""

from __future__ import annotations

from arrayff.potentials.eam.io.eam import EAMTable as EAMTable
from arrayff.potentials.eam.io.eam import read_setfl as read_setfl

__all__ = ["EAMTable", "read_setfl"]
