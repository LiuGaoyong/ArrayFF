# ArrayFF architecture

Status: initial framework (M0). Implemented: array dispatch, autodiff registry,
`Atoms` interop, neighbour tables, Lennard-Jones energy/forces.

## Decisions

| # | Decision |
|---|---|
| D1 | All kernel array operations go through `autoray.do`; kernels never import numpy/torch/jax. |
| D2 | Compilation is `ar.autojit` and only applies to *array level* functions. |
| D3 | `ase.Atoms` is the only structural input/output type. |
| D4 | Neighbour tables come from this package's sklearn backend, padded to a static `(N, M)` shape. |
| D5 | NumPy autodiff uses `advect` (not HIPS autograd). |
| D6 | QEq will detach charges (`stop_gradient`) before assembling forces. |
| D7 | Declared-but-unimplemented potentials raise `NotImplementedError` (`implemented=False` in the registry). |

## Layer map

```
ase.Atoms
  └─ utils/ase.py         Atoms -> positions, numbers, cell, pbc (backend arrays)
  └─ utils/neighbors.py   ase/neighbours -> padded NeighborTable (index, shift, mask)
      └─ potentials/*.py  energy_arrays(...) written only with do(...)
          └─ ad/          grad / value_and_grad / jacrev / jvp / vjp, backend resolved late
  └─ utils/dispatch.py    to_backend / to_numpy
```

The `Atoms` layer is always eager: JAX cannot trace Python objects, so `jit`
is applied to `energy_arrays` with precomputed, statically shaped tables.

## Autodiff

Transforms are registered in autoray's function registry
(`ar.register_function`) and resolved with `ar.get_lib_fn`, so adding a backend
means registering its factories:

| name | jax | torch | advect (numpy) |
|---|---|---|---|
| `grad` | `jax.grad` | `torch.func.grad` | `advect.grad` |
| `value_and_grad` | `jax.value_and_grad` | `torch.func.grad_and_value` (order swapped) | `advect.value_and_grad` |
| `jacrev` | `jax.jacrev` | `torch.func.jacrev` | `advect.jacobian` |
| `jvp` | immediate API | immediate API | transform factory (`tangents=`) |
| `vjp` | immediate API | immediate API | transform factory |

Plain numpy primals are routed to `advect` automatically.

## Verified interop facts (must stay true)

1. `do(...)` is transparent to JAX and torch autograd.
2. `ar.autojit(fn, backend="jax")` works for functions written with `do`.
3. autoray infers the backend name `advect` but has **no module mapping** for
   it: `authoray.do` inside a trace raises `ImportError` unless
   `register_module_alias("advect", "advect.numpy")` was called. This is done
   in `utils/dispatch.py` before every dispatch.
4. `advect` rejects `take` with a traced integer index (`TracingError`), so
   table lookups need a custom primitive or one-hot gather.
5. `ad.stop_gradient` truncates gradients inside a trace.
6. `torch.asarray` keeps `requires_grad` in torch >= 2.14, but older releases
   detached silently: keep the "gradient is not None" gate in tests.
7. `autoray`'s torch `take` requires an explicit `axis`, so kernels use
   advanced indexing (`sigma[species]`) instead.

## Testing policy

- `src/tests`: small systems (fcc Ar/Cu 2x2x2, Ar4 cluster, water dimer later),
  compared against ASE's official calculators or analytic values.
- `src/tests-benchmark`: several system sizes, `timeit.repeat` medians,
  compared across backends and against ASE implementations.
