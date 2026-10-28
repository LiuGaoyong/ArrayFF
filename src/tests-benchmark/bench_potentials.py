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
    # Positions change every MD step, so force a table rebuild (cache miss).
    def run():
        potential.clear_cache()
        potential.energy_and_forces(atoms)

    return _time(run)


def _jax_jit_case(potential, atoms):
    """Return (steady state seconds, compile seconds) for a jitted kernel."""
    import jax

    from arrayff.utils import jit

    context = potential._context(atoms)
    kernel = jit(potential.energy_arrays, backend="jax")

    start = timeit.default_timer()
    jax.block_until_ready(kernel(*context))
    compile_seconds = timeit.default_timer() - start

    def run():
        jax.block_until_ready(kernel(*context))

    return _time(run), compile_seconds


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
        for builder in ("vesin", "matscipy"):
            try:
                rows.append(
                    (
                        n,
                        "EMT",
                        f"numpy+{builder}",
                        None,
                        _arrayff_case(EMT(builder=builder), cu_atoms),
                        n,
                    )
                )
                rows.append(
                    (
                        n,
                        "LJ",
                        f"numpy+{builder}",
                        None,
                        _arrayff_case(
                            LennardJones(
                                cutoff=8.0,
                                sigma=3.405,
                                epsilon=0.01032,
                                shift=True,
                                builder=builder,
                            ),
                            ar_atoms,
                        ),
                        n,
                    )
                )
            except Exception:
                continue
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
        for backend in ("numpy", "torch"):
            try:
                calc_kernel = LennardJones(
                    cutoff=8.0,
                    sigma=3.405,
                    epsilon=0.01032,
                    shift=True,
                    backend=backend,
                )
                context = calc_kernel._context(ar_atoms)
                rows.append(
                    (
                        n,
                        "LJ",
                        f"{backend}-kernel",
                        None,
                        _time(
                            lambda ctx=context: calc_kernel.energy_arrays(*ctx)
                        ),
                        n,
                    )
                )
            except ImportError:
                pass

        try:
            jit_seconds, compile_seconds = _jax_jit_case(
                LennardJones(
                    cutoff=8.0,
                    sigma=3.405,
                    epsilon=0.01032,
                    shift=True,
                    backend="jax",
                ),
                ar_atoms,
            )
            rows.append((n, "LJ", "jax-jit", compile_seconds, jit_seconds, n))
        except ImportError:
            pass

        for backend in ("numpy", "torch"):
            try:
                calc_kernel = EMT(backend=backend)
                context = calc_kernel._context(cu_atoms)
                rows.append(
                    (
                        n,
                        "EMT",
                        f"{backend}-kernel",
                        None,
                        _time(
                            lambda ctx=context: calc_kernel.energy_arrays(*ctx)
                        ),
                        n,
                    )
                )
            except ImportError:
                pass

        try:
            jit_seconds, compile_seconds = _jax_jit_case(
                EMT(backend="jax"), cu_atoms
            )
            rows.append((n, "EMT", "jax-jit", compile_seconds, jit_seconds, n))
        except ImportError:
            pass
        assert lj is not None and emt is not None
    return rows


def _print_table(rows) -> None:
    header = (
        f"{'N':>6} {'potential':<10} {'impl':<14} {'ms':>10} "
        f"{'ms/atom':>10} {'compile ms':>11}"
    )
    print(header)
    print("-" * len(header))
    for n, potential, impl, compile_seconds, seconds, atoms in rows:
        ms = seconds * 1e3
        compile_ms = (
            "" if compile_seconds is None else f"{compile_seconds * 1e3:.0f}"
        )
        print(
            f"{n:>6} {potential:<10} {impl:<14} {ms:>10.3f} "
            f"{ms / atoms:>10.4f} {compile_ms:>11}"
        )


def test_benchmark_table():
    rows = _rows()
    _print_table(rows)
    print(
        "\nkernel rows (numpy-kernel/torch-kernel/jax-jit) time the "
        "array level "
        "kernels only: no neighbour search, no Atoms -> array conversion."
    )
    assert rows
    assert all(seconds > 0 for *_, seconds, _ in rows)


if __name__ == "__main__":
    _print_table(_rows())
    _ = np
