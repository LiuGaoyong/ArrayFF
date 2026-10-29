# ArrayFF

The Force Field Library Based on Multiple Array Backends like NumPy, PyTorch
and JAX.

Kernels are written once with [`autoray`](https://github.com/jcmgray/autoray),
so the backend is chosen by the arrays you pass in. Structural I/O uses
`ase.Atoms` throughout.

The only required array backends are NumPy and
[`advect`](https://github.com/jcmgray/advect) (for NumPy autodiff); PyTorch and
JAX are optional extras.

## Install

```bash
uv pip install -e .
uv pip install -e ".[torch]"     # optional PyTorch backend + nn wrappers
uv pip install -e ".[jax]"       # optional JAX backend
uv pip install -e ".[builders]"  # vesin / matscipy / pymatgen neighbours
```

## Quick start

```python
from ase.build import bulk
from arrayff.potentials import EMT, EMTCalculator

atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))

calc = EMT()                       # numpy backend + advect for forces
energy, forces = calc.energy_and_forces(atoms, method="analytic")

atoms.calc = EMTCalculator()       # ASE calculator interface
energy = atoms.get_potential_energy()
```

Backends are selected at construction:

```python
calc = EMT(backend="torch")        # or "jax"
calc = EMT(backend="jax", jit=True)
```

`LennardJones()` without arguments uses the OpenKIM universal shifted
parameters (118 elements, cutoff `4*sigma` per pair).

Each potential exposes analytic and autodiff forces/stress:

```python
from arrayff.potentials import LennardJones

calc = LennardJones(cutoff=8.0, sigma=3.405, epsilon=0.01032)
calc.energy_and_forces(atoms, method="analytic")
calc.energy_and_forces(atoms, method="autodiff")
calc.stress(atoms, method="analytic")
```

`jit=True` compiles the array-level kernel with `autoray.autojit` (JAX and
PyTorch are the useful backends; NumPy's compiler is best-effort).

## Charge equilibration and composite potentials

```python
from arrayff.electrostatics import QEq

qeq = QEq(chi={8: 8.5, 1: 4.5}, J={8: 10.0, 1: 12.0}, shielding=1.0)
charges = qeq.charges(atoms)       # sum == total charge

# additive composite: energy/forces/stress add, charges come from QEq
composite = LennardJones(cutoff=8.0, sigma=3.405, epsilon=0.01032) + qeq
energy, forces = composite.energy_and_forces(atoms)
```

QEq supports `charge_response="frozen"` (Hellmann-Feynman, default) and
`"relaxed"` (differentiate through the charge solve).

## Training (optional)

```python
from arrayff.potentials import LJTorch

model = LJTorch(species=[18])           # epsilon/sigma trainable
loss = (model(atoms) - target_energy) ** 2
loss.backward()
```

## Tests and benchmarks

```bash
pytest -q
pytest -s src/tests-benchmark/bench_potentials.py
pytest -s src/tests-benchmark/bench_derivatives.py
```

The benchmark tables compare analytic vs autodiff forces/stress and
`jit=False/True` across backends.

## Status

| Potential | State |
|---|---|
| Lennard-Jones (OpenKIM universal shifted) | implemented |
| Lennard-Jones 9-3, Morse | implemented |
| EMT (matches `ase.calculators.emt`, including `asap_cutoff`) | implemented |
| EAM (setfl tables, matches `ase.calculators.eam`) | implemented |
| QEq (shielded Coulomb, extended matrix, frozen/relaxed response) | implemented |
| Additive composites (`LJ + QEq`, ...) | implemented |
| MEAM, UF3, EEM, PQEq | declared placeholders |
