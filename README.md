# ArrayFF

The Force Field Library Based on Multiple Array Backends like NumPy, PyTorch
and JAX.

Kernels are written once with [`autoray`](https://github.com/jcmgray/autoray),
so the backend is chosen by the arrays you pass in. Structural I/O uses
`ase.Atoms` throughout.

## Install

```bash
uv pip install -e .
uv pip install -e ".[torch,jax,numpy-ad]"   # optional backends
```

## Quick start

```python
from ase.build import bulk
from arrayff.potentials import LennardJones, EMT, EAM   # noqa: F401
from arrayff.calculators import EMTCalculator

atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))

calc = EMT()                       # numpy backend, advect for forces
energy, forces = calc.energy_and_forces(atoms)

atoms.calc = EMTCalculator()       # ASE calculator interface
energy = atoms.get_potential_energy()
```

Backends are selected at construction:

```python
calc = EMT(backend="torch")        # or "jax"
```

`LennardJones()` without arguments uses the OpenKIM universal shifted
parameters (118 elements, cutoff `4*sigma` per pair).

## Charge equilibration

```python
from arrayff.electrostatics import QEq

model = QEq(chi={8: 8.5, 1: 4.5}, J={8: 10.0, 1: 12.0}, shielding=1.0)
charges = model.charges(atoms)          # sum == total charge
```

## Training (optional)

```python
from arrayff.nn import LJTorch

model = LJTorch(species=[18])           # epsilon/sigma trainable
loss = (model(atoms) - target_energy) ** 2
loss.backward()
```

## Tests and benchmarks

```bash
pytest -q                                  # unit tests (~15 s)
pytest -s src/tests-benchmark/bench_potentials.py   # timing table
```

## Status

| Potential | State |
|---|---|
| Lennard-Jones (OpenKIM universal shifted) | implemented |
| EMT (matches `ase.calculators.emt`) | implemented |
| EAM (setfl tables, matches `ase.calculators.eam`) | implemented |
| QEq (shielded Coulomb, extended matrix) | implemented |
| MEAM, UF3, EEM, PQEq | declared placeholders |
