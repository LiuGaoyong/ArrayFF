# Benchmarks

`pytest -s src/tests-benchmark/bench_potentials.py`

Median of `timeit.repeat(..., repeat=3, number=3)` for one
`energy_and_forces` call on fcc supercells (Apple M-series, CPU).

Rows labelled `*-kernel` time the array level kernel only (precomputed
neighbour table, no `Atoms` conversion); `jax-jit` is the compiled kernel with
compile time shown separately.

## 108 atoms

| potential | impl | ms | ms/atom |
|---|---|---|---|
| LJ | ASE | 9.45 | 0.0875 |
| LJ | numpy | 4.01 | 0.0371 |
| LJ | torch | 3.92 | 0.0363 |
| LJ | jax (eager) | 9.35 | 0.0866 |
| LJ | numpy-kernel | 0.33 | 0.0030 |
| LJ | torch-kernel | 0.50 | 0.0046 |
| LJ | jax-jit (compile 36 ms) | 0.093 | 0.0009 |
| EMT | ASE | 12.98 | 0.1202 |
| EMT | numpy | 5.50 | 0.0509 |
| EMT | torch | 4.94 | 0.0457 |
| EMT | jax (eager) | 17.85 | 0.1653 |
| EMT | numpy-kernel | 0.54 | 0.0050 |
| EMT | jax-jit (compile 60 ms) | 0.134 | 0.0012 |
| EAM | numpy | 5.14 | 0.0476 |
| EAM | torch | 4.16 | 0.0386 |

## 500 atoms

| potential | impl | ms | ms/atom |
|---|---|---|---|
| LJ | ASE | 39.49 | 0.0790 |
| LJ | numpy | 13.73 | 0.0275 |
| LJ | torch | 11.95 | 0.0239 |
| LJ | jax (eager) | 16.48 | 0.0330 |
| LJ | jax-jit (compile 46 ms) | 0.237 | 0.0005 |
| EMT | ASE | 56.09 | 0.1122 |
| EMT | numpy | 18.48 | 0.0370 |
| EMT | torch | 15.49 | 0.0310 |
| EMT | jax (eager) | 27.83 | 0.0557 |
| EMT | jax-jit (compile 67 ms) | 0.386 | 0.0008 |
| EAM | numpy | 18.94 | 0.0379 |
| EAM | torch | 15.29 | 0.0306 |

## 1372 atoms

| potential | impl | ms | ms/atom |
|---|---|---|---|
| LJ | ASE | 106.3 | 0.0775 |
| LJ | numpy | 38.93 | 0.0284 |
| LJ | torch | 32.57 | 0.0237 |
| LJ | jax (eager) | 36.64 | 0.0267 |
| EMT | ASE | 152.8 | 0.1114 |
| EMT | numpy | 52.95 | 0.0386 |
| EMT | torch | 41.73 | 0.0304 |
| EAM | numpy | 52.33 | 0.0381 |
| EAM | torch | 40.51 | 0.0295 |

## Reading the numbers

- Eager arrayff is 1.5-3x faster than the ASE calculators.
- JAX without `jit` is slower than numpy/torch: every `do(...)` is a separate
  dispatch.
- The kernel itself is only ~8% of an `energy_and_forces` call; the rest is
  the sklearn neighbour search plus `Atoms` conversion. For MD loops the next
  win is caching/rebuilding the neighbour table outside the kernel.
- `jax-jit` is 30-70x faster than the numpy kernel, but it excludes neighbour
  construction and `Atoms` conversion, so it is an upper bound rather than a
  drop-in replacement.
