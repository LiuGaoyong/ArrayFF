"""File readers for tabulated potentials."""

from __future__ import annotations

from arrayff.io.eam import EAMTable as EAMTable
from arrayff.io.eam import read_setfl as read_setfl

__all__ = ["EAMTable", "read_setfl"]
