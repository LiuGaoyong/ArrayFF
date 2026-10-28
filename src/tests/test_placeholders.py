"""Declared but unimplemented models must fail loudly, not silently."""

from __future__ import annotations

import pytest

from arrayff import available_potentials, get_potential
from arrayff.electrostatics import EEM, PQEq
from arrayff.potentials import MEAM, UF3
from arrayff.utils.exceptions import ScopeError


def test_registry_marks_placeholders():
    status = available_potentials()
    assert status["meam"] is False
    assert status["uf3"] is False
    assert status["lj"] is True


@pytest.mark.parametrize("name", ["meam", "uf3"])
def test_get_potential_rejects_placeholders(name):
    with pytest.raises(NotImplementedError, match=name):
        get_potential(name)


@pytest.mark.parametrize("cls", [MEAM, UF3])
def test_placeholder_potentials_raise_scope_error(cls):
    with pytest.raises(ScopeError):
        cls()


@pytest.mark.parametrize("cls", [EEM, PQEq])
def test_placeholder_charge_models_raise_scope_error(cls):
    with pytest.raises(ScopeError):
        cls()
