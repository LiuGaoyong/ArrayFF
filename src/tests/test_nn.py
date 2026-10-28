"""Torch modules (optional dependency)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ase.build import bulk  # noqa: E402

from arrayff.nn import EMTTorch, LJTorch  # noqa: E402
from arrayff.potentials import EMT, LennardJones  # noqa: E402


def _ar():
    return bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))


def test_lj_torch_matches_kim_parameters():
    atoms = _ar()
    model = LJTorch(species=[18])
    reference = LennardJones()
    assert model(atoms).detach().item() == pytest.approx(
        reference.energy(atoms), rel=1e-12
    )


def test_lj_torch_parameters_are_trainable():
    model = LJTorch(species=[18])
    atoms = _ar()
    energy = model(atoms)
    loss = (energy - (-5.0)) ** 2
    loss.backward()
    assert model.raw_epsilon.grad is not None
    assert model.raw_sigma.grad is not None
    assert float(model.raw_epsilon.grad.abs()) > 0.0


def test_lj_torch_training_step_reduces_loss():
    atoms = _ar()
    model = LJTorch(species=[18])
    target = -1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = None
    for _ in range(20):
        optimizer.zero_grad()
        loss = (model(atoms) - target) ** 2
        loss.backward()
        optimizer.step()
        if initial is None:
            initial = loss.detach().item()
    assert loss.detach().item() < initial


def test_emt_torch_matches_emt():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    assert EMTTorch()(atoms).detach().item() == pytest.approx(
        EMT().energy(atoms), rel=1e-12
    )


def test_forces_tensor_shape():
    model = LJTorch(species=[18])
    forces = model.forces_tensor(_ar())
    assert tuple(forces.shape) == (32, 3)
