"""jit on/off must agree; unsupported backends fall back safely."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from arrayff.potentials import EMT, LennardJones
from arrayff.utils import enable_x64


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("method", ["analytic", "autodiff"])
def test_lj_jit_matches_eager(backend, method):
    pytest.importorskip(backend)
    if backend == "jax":
        enable_x64()
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    kwargs = dict(cutoff=8.0, sigma=3.405, epsilon=0.01032, shift=True)
    eager = LennardJones(backend=backend, method=method, **kwargs)
    jitted = LennardJones(backend=backend, method=method, jit=True, **kwargs)
    a = eager.energy_and_forces(atoms)
    b = jitted.energy_and_forces(atoms)
    assert b[0] == pytest.approx(a[0], rel=1e-12)
    np.testing.assert_allclose(b[1], a[1], atol=1e-9)
    np.testing.assert_allclose(
        jitted.stress(atoms) if method == "analytic" else eager.stress(atoms),
        eager.stress(atoms),
        atol=1e-8,
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_emt_jit_matches_eager(backend):
    pytest.importorskip(backend)
    if backend == "jax":
        enable_x64()
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    eager = EMT(backend=backend)
    jitted = EMT(backend=backend, jit=True)
    a = eager.energy_and_forces(atoms)
    b = jitted.energy_and_forces(atoms)
    assert b[0] == pytest.approx(a[0], rel=1e-12)
    np.testing.assert_allclose(b[1], a[1], atol=1e-9)
    np.testing.assert_allclose(
        jitted.stress(atoms),
        eager.stress(atoms),
        atol=1e-8,
    )
