"""Tests for arrayff.optimize submodule."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.lj import LennardJones

from arrayff.optimize import (
    DimerResult,
    NEBResult,
    OptimizationResult,
    call_dimer,
    call_dimer_displace,
    call_neb,
    call_optimization,
    call_vib,
)
from arrayff.optimize._neb import call_neb_raw

SIGMA = 3.405
EPSILON = 0.01032
CUTOFF = 10.0


def _lj_calc():
    """Create an ASE Lennard-Jones calculator."""
    return LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF)


def _ar_cluster():
    """Create a small Ar cluster."""
    return Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [0.0, 3.5, 0.0],
            [3.5, 3.5, 0.0],
        ],
        cell=None,
        pbc=False,
    )


def test_optimization_result_basic():
    """Test OptimizationResult dataclass."""
    atoms = _ar_cluster()
    result = OptimizationResult(lst=[atoms], converged=True)
    assert result.converged is True
    assert len(result.lst) == 1
    assert result.latest is atoms


def test_call_optimization_basic():
    """Test basic structure optimization."""
    atoms = _ar_cluster()
    calc = _lj_calc()
    result = call_optimization(
        atoms, calc, method="LBFGS", max_steps=50, fmax=0.01
    )
    assert isinstance(result, OptimizationResult)
    assert len(result.lst) > 0
    # After optimization, forces should be small
    optimized = result.latest
    forces = optimized.get_forces()
    assert np.linalg.norm(forces, axis=1).max() < 0.1


def test_call_optimization_methods():
    """Test that different optimizer methods work."""
    atoms = _ar_cluster()
    calc = _lj_calc()
    for method in ["LBFGS", "BFGS", "FIRE"]:
        atoms_copy = atoms.copy()
        result = call_optimization(
            atoms_copy, calc, method=method, max_steps=50, fmax=0.05
        )
        assert isinstance(result, OptimizationResult)
        assert len(result.lst) > 0


def test_call_optimization_invalid_method():
    """Test that invalid method raises assertion."""
    atoms = _ar_cluster()
    calc = _lj_calc()
    with pytest.raises(AssertionError):
        call_optimization(atoms, calc, method="InvalidMethod", max_steps=10)


def test_call_vib():
    """Test vibration analysis."""
    from ase.calculators.emt import EMT

    atoms = molecule("H2O")
    calc = EMT()
    # Optimize first to avoid imaginary frequencies
    result = call_optimization(
        atoms, calc, method="LBFGS", max_steps=50, fmax=0.01
    )
    optimized = result.latest
    freq, modes = call_vib(optimized.copy(), calc)
    assert freq.shape[0] == 9  # 3N = 3*3 = 9 for H2O
    assert modes.shape == (9, 3, 3)
    # Six frequencies should be near zero (translations/rotations)
    # After ignore_fqmin=10, they should be set to 1e-5
    near_zero = np.abs(freq) < 100
    assert near_zero.sum() >= 5  # At least translations + rotations


def test_call_dimer_basic():
    """Test basic dimer calculation."""
    # Use a simple double-well potential scenario
    atoms = _ar_cluster()
    atoms.positions[0] += 0.5  # Displace one atom to create non-equilibrium
    calc = _lj_calc()
    result = call_dimer(atoms, calc, max_steps=5, fmax=0.1, logfile="-")
    assert isinstance(result, DimerResult)
    assert len(result.lst) >= 2  # Initial + at least one step
    assert result.lst[0].info.get("dimer_curvature") is not None


def test_call_dimer_displace():
    """Test call_dimer_displace returns displacement vector."""
    atoms = _ar_cluster()
    calc = _lj_calc()
    displacement = call_dimer_displace(atoms, calc, logfile="-")
    assert displacement.shape == (4, 3)  # 4 atoms, 3 dimensions


def test_call_neb_basic():
    """Test basic NEB calculation."""
    # Create initial and final states by displacing one atom
    initial = _ar_cluster()
    final = _ar_cluster()
    final.positions[0] += 2.0  # Move first atom by 2 Angstrom

    calc = _lj_calc()
    # Use few images and steps for quick test
    result = call_neb(
        initial,
        calc,
        final,
        nimages=3,
        climb=False,
        max_steps=10,
        fmax=0.1,
        method4interpolate="linear",
    )
    assert isinstance(result, NEBResult)
    assert len(result.lst) > 0


def test_call_neb_raw():
    """Test NEB with raw image list."""
    images = [_ar_cluster() for _ in range(5)]
    # Create path by linearly interpolating positions
    start = _ar_cluster().positions
    end = _ar_cluster().positions.copy()
    end[0] += 2.0
    for i, img in enumerate(images):
        alpha = i / (len(images) - 1)
        img.positions = start * (1 - alpha) + end * alpha

    calc = _lj_calc()
    result = call_neb_raw(
        images,
        calc,
        climb=False,
        max_steps=5,
        fmax=0.2,
        method4spring="improvedtangent",
    )
    assert isinstance(result, OptimizationResult)
    assert len(result.lst) > 0
