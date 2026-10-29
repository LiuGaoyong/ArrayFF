"""Tests for the autoray registered autodiff transforms."""

from __future__ import annotations

import numpy as np
import pytest

from arrayff.utils.ad import (
    grad,
    jacrev,
    jvp,
    registered_backends,
    stop_gradient,
    value_and_grad,
    vjp,
)
from arrayff.utils.dispatch import do

F = lambda x: do("sum", x**2)  # noqa: E731


def test_numpy_routes_to_advect():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(grad(F)(x), 2 * x)
    value, gradient = value_and_grad(F)(x)
    assert float(value) == pytest.approx(14.0)
    np.testing.assert_allclose(gradient, 2 * x)
    assert "advect" in registered_backends()


def test_numpy_jacrev():
    jacobian = np.asarray(jacrev(lambda x: x**2)(np.array([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(jacobian, np.diag([2.0, 4.0, 6.0]))


def test_torch_transforms():
    torch = pytest.importorskip("torch")
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    np.testing.assert_allclose(grad(F)(x).detach().numpy(), 2 * x.numpy())

    # value comes first, matching the JAX convention
    value, gradient = value_and_grad(F)(x)
    assert float(value) == pytest.approx(14.0)
    np.testing.assert_allclose(gradient.detach().numpy(), 2 * x.numpy())

    jacobian = jacrev(lambda v: v**2)(x.to(torch.float64))
    np.testing.assert_allclose(jacobian.numpy(), np.diag([2.0, 4.0, 6.0]))


def test_jax_transforms():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    x = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(grad(F)(x)), 2 * np.asarray(x))

    value, tangent = jvp(F)((x,), (jnp.ones(3),))
    assert float(value) == pytest.approx(14.0)
    assert float(tangent) == pytest.approx(12.0)

    value, pullback = vjp(F)((x,))
    (gradient,) = pullback(jnp.array(1.0))
    assert float(value) == pytest.approx(14.0)
    np.testing.assert_allclose(np.asarray(gradient), 2 * np.asarray(x))


def test_stop_gradient_cuts_the_tape():
    x = np.array([1.0, 2.0])
    detached = stop_gradient(x)
    np.testing.assert_allclose(detached, x)

    def mixed(v):
        return do("sum", v) * stop_gradient(do("sum", v))

    gradient = grad(mixed)(x)
    np.testing.assert_allclose(gradient, [3.0, 3.0])


def test_stop_gradient_torch():
    torch = pytest.importorskip("torch")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = stop_gradient(x * 2.0)
    assert not y.requires_grad


def test_mixed_backends_raise():
    torch = pytest.importorskip("torch")
    from arrayff.utils import require_same_backend

    with pytest.raises(TypeError, match="backend"):
        require_same_backend(np.ones(2), torch.ones(2))
