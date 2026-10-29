"""Low level utilities shared by the rest of :mod:`arrayff`."""

from __future__ import annotations

from arrayff.utils.backend import (
    ARRAY_BACKENDS,
    available_backends,
    default_dtype,
    infer_backend,
    resolve_ad_backend,
)
from arrayff.utils.dispatch import (
    do,
    dtype_of,
    register_autograd_alias,
    require_same_backend,
    to_backend,
    to_numpy,
)
from arrayff.utils.exceptions import (
    ArrayFFError,
    MissingBackendError,
    ScopeError,
    ShapeError,
)
from arrayff.utils.jit import enable_x64, jit, x64_enabled
from arrayff.utils.optional import is_available, require

__all__ = [
    "ARRAY_BACKENDS",
    "ArrayFFError",
    "MissingBackendError",
    "ScopeError",
    "ShapeError",
    "available_backends",
    "default_dtype",
    "do",
    "dtype_of",
    "enable_x64",
    "infer_backend",
    "is_available",
    "jit",
    "require",
    "register_autograd_alias",
    "require_same_backend",
    "resolve_ad_backend",
    "to_backend",
    "to_numpy",
    "x64_enabled",
]
