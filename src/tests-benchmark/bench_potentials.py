"""Timing of the potentials across backends, methods and jit settings.

Run with::

    pixi run benchmark
    # or
    python -m pytest -s src/tests-benchmark/bench_potentials.py

The table reports the median of ``timeit.repeat`` for one
``energy_and_forces`` call.  QEq (dense solve) and the LJ+QEq composite use
smaller cells (Q14=A).
"""

from __future__ import annotations

import contextlib
import io
import statistics
import tempfile
import timeit
import warnings
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT as AseEMT
from ase.calculators.lj import LennardJones as AseLJ

from arrayff.electrostatics import QEq
from arrayff.potentials import EAM, EMT, LennardJones
from arrayff.utils import enable_x64

enable_x64()
warnings.simplefilter("ignore")

SIZES = (3, 5, 7)  # fcc repetitions -> 108 / 500 / 1372 atoms
QEQ_SIZES = ((2, 2, 2), (2, 2, 4), (3, 3, 3))  # 32 / 64 / 108
REPEAT = 2
NUMBER = 2
BACKENDS = ("numpy", "torch", "jax")


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


def _ase_case(name: str, atoms) -> float:
    if name == "lj":
        atoms.calc = AseLJ(sigma=3.405, epsilon=0.01032, rc=8.0)
    else:
        atoms.calc = AseEMT()

    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            atoms.calc.calculate(atoms)

    return _time(run)


def _arrayff_case(potential, atoms) -> tuple[float, float | None]:
    compile_seconds = None
    if getattr(potential, "jit", False):
        start = timeit.default_timer()
        potential.energy_and_forces(atoms)
        compile_seconds = timeit.default_timer() - start

    def run():
        potential.clear_cache()
        potential.energy_and_forces(atoms)

    return _time(run), compile_seconds


def _row(n, name, impl, method, jit, seconds, compile_seconds):
    return (n, name, impl, method, jit, seconds, compile_seconds)


def _rows():
    rows = []
    eam_tmp = tempfile.TemporaryDirectory()
    eam_table = _write_eam_table(Path(eam_tmp.name) / "Cu.eam.alloy")
    try:
        for repeat in SIZES:
            ar_atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(
                (repeat,) * 3
            )
            cu_atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(
                (repeat,) * 3
            )
            n = len(cu_atoms)
            rows.append(
                _row(
                    n, "LJ", "ASE", "-", False,
                    _ase_case("lj", ar_atoms), None,
                )
            )
            rows.append(
                _row(
                    n, "EMT", "ASE", "-", False,
                    _ase_case("emt", cu_atoms), None,
                )
            )
            for backend in BACKENDS:
                for method in ("analytic", "autodiff"):
                    try:
                        potential = LennardJones(
                            cutoff=8.0,
                            sigma=3.405,
                            epsilon=0.01032,
                            shift=True,
                            backend=backend,
                            method=method,
                        )
                        rows.append(
                            _row(
                                n,
                                "LJ",
                                backend,
                                method,
                                False,
                                *_arrayff_case(potential, ar_atoms),
                            )
                        )
                        jitted = LennardJones(
                            cutoff=8.0,
                            sigma=3.405,
                            epsilon=0.01032,
                            shift=True,
                            backend=backend,
                            method=method,
                            jit=True,
                        )
                        seconds, compile_seconds = _arrayff_case(
                            jitted, ar_atoms
                        )
                        rows.append(
                            _row(
                                n,
                                "LJ",
                                backend,
                                method,
                                True,
                                seconds,
                                compile_seconds,
                            )
                        )
                    except ImportError:
                        continue
                try:
                    potential = EMT(backend=backend)
                    rows.append(
                        _row(
                            n,
                            "EMT",
                            backend,
                            "analytic",
                            False,
                            *_arrayff_case(potential, cu_atoms),
                        )
                    )
                    jitted = EMT(backend=backend, jit=True)
                    seconds, compile_seconds = _arrayff_case(jitted, cu_atoms)
                    rows.append(
                        _row(
                            n, "EMT", backend, "analytic", True, seconds,
                            compile_seconds,
                        )
                    )
                    eam = EAM(str(eam_table), backend=backend)
                    rows.append(
                        _row(
                            n,
                            "EAM",
                            backend,
                            "analytic",
                            False,
                            *_arrayff_case(eam, cu_atoms),
                        )
                    )
                    eam_jit = EAM(str(eam_table), backend=backend, jit=True)
                    seconds, compile_seconds = _arrayff_case(eam_jit, cu_atoms)
                    rows.append(
                        _row(
                            n, "EAM", backend, "analytic", True, seconds,
                            compile_seconds,
                        )
                    )
                except ImportError:
                    continue
    finally:
        eam_tmp.cleanup()

    # QEq and the additive composite use smaller cells (dense O(N^3) solve).
    for repeats in QEQ_SIZES:
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(repeats)
        n = len(atoms)
        for backend in BACKENDS:
            try:
                qeq = QEq(chi=4.5, J=12.0, backend=backend)
                rows.append(
                    _row(
                        n,
                        "QEq",
                        backend,
                        "analytic",
                        False,
                        *_arrayff_case(qeq, atoms),
                    )
                )
                qeq_jit = QEq(chi=4.5, J=12.0, backend=backend, jit=True)
                seconds, compile_seconds = _arrayff_case(qeq_jit, atoms)
                rows.append(
                    _row(n, "QEq", backend, "analytic", True, seconds,
                         compile_seconds)
                )
            except ImportError:
                continue
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    for backend in BACKENDS:
        try:
            composite = LennardJones(
                cutoff=8.0, sigma=3.405, epsilon=0.01032,
                backend=backend,
            ) + QEq(chi=4.5, J=12.0, backend=backend)
            rows.append(
                _row(
                    len(atoms),
                    "LJ+QEq",
                    backend,
                    "analytic",
                    False,
                    *_arrayff_case(composite, atoms),
                )
            )
        except ImportError:
            continue
    return rows


def _print_table(rows) -> None:
    header = (
        f"{'N':>6} {'potential':<8} {'backend':<8} {'method':<10} "
        f"{'jit':<5} {'ms':>10} {'ms/atom':>10} {'compile ms':>11}"
    )
    print(header)
    print("-" * len(header))
    for n, name, backend, method, jit, seconds, compile_seconds in rows:
        ms = seconds * 1e3
        compile_ms = (
            "" if compile_seconds is None else f"{compile_seconds * 1e3:.0f}"
        )
        print(
            f"{n:>6} {name:<8} {backend:<8} {method:<10} {str(jit):<5} "
            f"{ms:>10.3f} {ms / n:>10.4f} {compile_ms:>11}"
        )


def test_benchmark_table():
    rows = _rows()
    _print_table(rows)
    assert rows
    assert all(seconds > 0 for *_, seconds, _ in rows)


if __name__ == "__main__":
    _print_table(_rows())
