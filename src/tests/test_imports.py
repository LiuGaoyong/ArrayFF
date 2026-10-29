"""Import purity: optional backends must not load on ``import arrayff``."""

from __future__ import annotations

import subprocess
import sys

import pytest

from arrayff.utils.optional import is_available


def test_core_import_does_not_load_optional_backends():
    code = (
        "import sys, arrayff, arrayff.potentials, arrayff.neighbors; "
        "bad = [m for m in ('torch', 'jax') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_optional_modules_are_importable():
    # Importing the modules themselves must not fail at package import time.
    import arrayff
    import arrayff.potentials
    import arrayff.neighbors
    import arrayff.utils

    assert arrayff.__version__


@pytest.mark.skipif(
    not is_available("torch"),
    reason="torch not installed",
)
def test_torch_modules_are_opt_in():
    code = (
        "import sys, arrayff; "
        "assert 'torch' not in sys.modules; "
        "from arrayff.potentials import LJTorch; "
        "assert 'torch' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
