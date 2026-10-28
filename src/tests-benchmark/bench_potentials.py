"""Timing of the potentials across backends and system sizes.

Run with::

    pixi run benchmark
    # or
    python -m pytest -s src/tests-benchmark/bench_potentials.py

The table reports the median of ``timeit.repeat`` for a single
``energy_and_forces`` call, plus ms/atom and the speedup over ASE.
"""

from __future__ import annotations

import contextlib
import io
import statistics
import tempfile
import timeit
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT as AseEMT
from ase.calculators.lj import LennardJones as AseLJ

from arrayff.potentials import EAM, EMT, LennardJones

SIZES = (3, 5, 7)  # fcc repetitions -> 108 / 500 / 1372 atoms
REPEAT = 3
NUMBER = 3


def _time(fn) -> float:
    return (
        statistics.median(timeit.repeat(fn, repeat=REPEAT, number=NUMBER))
        / NUMBER
    )


def _write_eam_table(path: Path) -> Path:
    nrho, drho, nr, dr, cutoff = 2000, 0.01, 6000, 0.001, 6.0
    rho = np.arange(nrho) * drho
    r = np.arange(nr) * dr
    embedded = -np.sqrt(rho) + 0.1 * rho
    density = 10.0 * np.clip(1.0 - r / cutoff, 0.0, None) ** 4
    rphi = np.zeros(nr)
    rphi[1:] = 2.0 * (2.5 / r[1:]) ** 6 - 0.5 * (2.5 / r[1:]) ** 3
    with open(path, "w") as handle:
        handle.write("benchmark Cu\n\narrayff\n1\nCu\n")
        handle.write(f"{nrho} {drho} {nr} {dr} {cutoff}\n")
        handle.write("29 63.546 3.615 FCC\n")
        for values in (embedded, density, rphi):
            handle.write("\n".join(f"{v:.10e}" for v in values) + "\n")
    return path


def _case(symbol: str, lattice: float, repeat: int):
    atoms = bulk(symbol, "fcc", a=lattice, cubic=True).repeat((repeat,) * 3)
    return atoms


def _ase_case(name: str, atoms):
    if name == "lj":
        atoms.calc = AseLJ(sigma=3.405, epsilon=0.01032, rc=8.0)
    elif name == "emt":
        atoms.calc = AseEMT()
    else:
        atoms.calc = AseEMT()

    # Force a real recalculation: ASE caches results between calls. Some ASE
    # calculators print during calculate, so silence them.
    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            atoms.calc.calculate(atoms)

    return _time(run)


def _arrayff_case(potential, atoms):
    return _time(lambda: potential.energy_and_forces(atoms))


def _rows():
    rows = []
    for repeat in SIZES:
        ar_atoms = _case("Ar", 5.26, repeat)
        cu_atoms = _case("Cu", 3.6, repeat)
        n = len(cu_atoms)

        lj = LennardJones(cutoff=8.0, sigma=3.405, epsilon=0.01032, shift=True)
        emt = EMT()
        rows.append((n, "LJ", "ASE", None, _ase_case("lj", ar_atoms), n))
        for backend in ("numpy", "torch", "jax"):
            try:
                calc = LennardJones(
                    cutoff=8.0,
                    sigma=3.405,
                    epsilon=0.01032,
                    shift=True,
                    backend=backend,
                )
                rows.append(
                    (n, "LJ", backend, None, _arrayff_case(calc, ar_atoms), n)
                )
            except ImportError:
                continue
        rows.append((n, "EMT", "ASE", None, _ase_case("emt", cu_atoms), n))
        for backend in ("numpy", "torch", "jax"):
            try:
                rows.append(
                    (
                        n,
                        "EMT",
                        backend,
                        None,
                        _arrayff_case(EMT(backend=backend), cu_atoms),
                        n,
                    )
                )
            except ImportError:
                continue

        with tempfile.TemporaryDirectory() as tmp:
            table = _write_eam_table(Path(tmp) / "Cu.eam.alloy")
            for backend in ("numpy", "torch"):
                rows.append(
                    (
                        n,
                        "EAM",
                        backend,
                        None,
                        _arrayff_case(
                            EAM(str(table), backend=backend), cu_atoms
                        ),
                        n,
                    )
                )
        assert lj is not None and emt is not None
    return rows


def _print_table(rows) -> None:
    header = (
        f"{'N':>6} {'potential':<10} {'impl':<14} {'ms':>10} {'ms/atom':>10}"
    )
    print(header)
    print("-" * len(header))
    for n, potential, impl, _, seconds, atoms in rows:
        ms = seconds * 1e3
        print(
            f"{n:>6} {potential:<10} {impl:<14} {ms:>10.3f} {ms / atoms:>10.4f}"
        )


def test_benchmark_table():
    rows = _rows()
    _print_table(rows)
    assert rows
    assert all(seconds > 0 for *_, seconds, _ in rows)


if __name__ == "__main__":
    _print_table(_rows())
    _ = np
