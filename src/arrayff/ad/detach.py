"""Detach values from the autodiff tape."""

from __future__ import annotations

from typing import Any

from arrayff.utils.dispatch import do


def stop_gradient(x: Any) -> Any:
    """Return ``x`` with its gradient contribution removed.

    Dispatches through autoray, so the same call works for numpy (identity),
    torch (``.detach()``), JAX (``lax.stop_gradient``) and advect.
    """
    return do("stop_gradient", x)
