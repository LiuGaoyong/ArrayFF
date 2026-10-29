"""Compare analytic and autodiff forces/stress per backend.

Prints a markdown table with timings and the maximum analytic-vs-AD
difference.  This is deliberately small (108-atom systems) so it can run on
a laptop and in CI.
"""

from __future__ import annotations

import statistics
import tempfile
import timeit
import warnings
from pathlib import Path

import numpy as np
from ase.build import bulk

from arrayff.electrostatics import QEq
from arrayff.potentials import EAM, EMT, LennardJones
from arrayff.utils import enable_x64

enable_x64()
warnings.simplefilter("ignore")

REPEAT = 2
NUMBER = 2
BACKENDS = ("numpy", "torch", "jax")


def _time(fn) -> float:
    return statistics.median(
        timeit.repeat(fn, repeat=REPEAT, number=NUMBER)
    ) / NUMBER


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


def _case(symbol: str, lattice: float):
    return bulk(symbol, "fcc", a=lattice, cubic=True).repeat((3, 3, 3))


def _measure(potential, atoms):
    def forces():
        potential.clear_cache()
        potential.energy_and_forces(atoms)

    def stress():
        potential.clear_cache()
        potential.stress(atoms)

    return _time(forces), _time(stress)


def _rows():
    ar_atoms = _case("Ar", 5.26)
    cu_atoms = _case("Cu", 3.6)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        eam_table = _write_eam_table(Path(tmp) / "Cu.eam.alloy")
        for backend in BACKENDS:
            try:
                factories = [
                    (
                        "LJ",
                        ar_atoms,
                        lambda b=backend: LennardJones(
                            cutoff=8.0,
                            sigma=3.405,
                            epsilon=0.01032,
                            shift=True,
                            smooth=True,
                            backend=b,
                        ),
                    ),
                    ("EMT", cu_atoms, lambda b=backend: EMT(backend=b)),
                    (
                        "EAM",
                        cu_atoms,
                        lambda b=backend: EAM(str(eam_table), backend=b),
                    ),
                    (
                        "QEq",
                        _case("Cu", 3.6),
                        lambda b=backend: QEq(
                            chi=4.5, J=12.0, backend=b
                        ),
                    ),
                ]
                for name, atoms, factory in factories:
                    analytic = factory()
                    analytic.method = "analytic"
                    autodiff = factory()
                    autodiff.method = "autodiff"
                    analytic.energy_and_forces(atoms, method="analytic")
                    autodiff.energy_and_forces(atoms, method="autodiff")
                    f_analytic, s_analytic = _measure(analytic, atoms)
                    f_autodiff, s_autodiff = _measure(autodiff, atoms)
                    forces_analytic = analytic.energy_and_forces(
                        atoms, method="analytic"
                    )[1]
                    forces_autodiff = autodiff.energy_and_forces(
                        atoms, method="autodiff"
                    )[1]
                    stress_analytic = analytic.stress(atoms, method="analytic")
                    stress_autodiff = autodiff.stress(atoms, method="autodiff")
                    rows.append(
                        (
                            name,
                            backend,
                            f_analytic,
                            f_autodiff,
                            s_analytic,
                            s_autodiff,
                            np.max(
                                np.abs(forces_analytic - forces_autodiff)
                            ),
                            np.max(
                                np.abs(stress_analytic - stress_autodiff)
                            ),
                        )
                    )
            except ImportError:
                continue
    return rows


def _print_table(rows) -> None:
    header = (
        "| potential | backend | force analytic ms | force AD ms | "
        "stress analytic ms | stress AD ms | max ΔF | max Δσ |"
    )
    print(header)
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, backend, fa, fd, sa, sd, df, ds in rows:
        print(
            f"| {name} | {backend} | {fa * 1e3:.3f} | {fd * 1e3:.3f} | "
            f"{sa * 1e3:.3f} | {sd * 1e3:.3f} | {df:.2e} | {ds:.2e} |"
        )


def test_derivative_table():
    rows = _rows()
    _print_table(rows)
    assert rows
    assert all(fa > 0 and fd > 0 for _, _, fa, fd, *_ in rows)
    assert max(row[6] for row in rows) < 1e-7
    assert max(row[7] for row in rows) < 1e-7


if __name__ == "__main__":
    _print_table(_rows())
