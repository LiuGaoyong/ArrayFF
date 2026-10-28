"""Exceptions raised by :mod:`arrayff`."""

from __future__ import annotations


class ArrayFFError(Exception):
    """Base class for all arrayff specific errors."""


class MissingBackendError(ArrayFFError, ImportError):
    """A requested array backend (or extra) is not installed."""


class ShapeError(ArrayFFError, ValueError):
    """An array has an unexpected shape or dtype."""


class ScopeError(ArrayFFError, NotImplementedError):
    """A feature is declared in the API but not implemented yet."""
