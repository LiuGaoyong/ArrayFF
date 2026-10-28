"""Automatic differentiation that follows the array backend."""

from __future__ import annotations

from arrayff.ad.detach import stop_gradient
from arrayff.ad.registry import register_backend, registered_backends
from arrayff.ad.transforms import grad, jacrev, jvp, value_and_grad, vjp

__all__ = [
    "grad",
    "jacrev",
    "jvp",
    "register_backend",
    "registered_backends",
    "stop_gradient",
    "value_and_grad",
    "vjp",
]
