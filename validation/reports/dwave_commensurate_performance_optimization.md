# d-wave commensurate Ward performance optimization

## Scope

The public commands

```bash
python -m validation ward bond-metric-full-kernel
python -m validation ward bond-metric-family
```

now use optimized validation-only execution while preserving the established
48-component primitive vector, compensated periodic sum, counterterm policy,
Schur assembly, Ward audit, output schema, and fail-closed status.

## Implemented optimizations

### Batched chunk diagonalization

The reference pointwise evaluator called `np.linalg.eigh` separately for the
midpoint, minus endpoint, and plus endpoint of every k point.  The optimized
fallback evaluator assembles each chunk and performs three stacked Hermitian
diagonalizations.

### Exact commensurate eigensystem cache

For

\[
q = \frac{2\pi}{N_k}(m_x,m_y),
\]

`k +/- q/2` belongs to the same shifted tensor subgrid when both integer
components are even, or to a known complementary half-step subgrid otherwise.
The optimized full-kernel command therefore:

1. diagonalizes each required subgrid once;
2. stores energies, states, and occupations in lexicographic index order;
3. selects midpoint/minus/plus bands with exact periodic integer index maps;
4. performs the unchanged primitive contractions and compensated integration.

The eigensystem count for one point changes from

```text
3 * subgrid_count * Nk^2
```

to

```text
subgrid_count * Nk^2
```

while the number of primitive point evaluations remains unchanged.

### Reused midpoint thermal density

At each midpoint, the optimized evaluator constructs

\[
\rho_T = \frac12 U f(E) U^\dagger
\]

once.  Contact, phase-direct, and Ward `delta_v` thermal expectations then use
`Tr(rho_T V)` instead of repeating a complete band-basis transformation.

### Fixed contractions

The unified five-channel bubble and equal-time Ward term use fixed reshape/GEMM
contractions instead of requesting a new dynamic `einsum` path at every point.

### Family-level concurrency

`bond-metric-family` accepts

```bash
--workers N
```

Independent q points run as separate subprocesses.  BLAS should remain
single-threaded per worker.  Two or three workers are the recommended first test
on the current laptop; higher values should be chosen only after observing peak
memory and CPU saturation.

## Numerical verification

Tests cover:

- optimized chunk evaluator versus the reference pointwise 48-component vector;
- exact even-half-q cache indexing;
- odd-half-q complementary-subgrid cache indexing;
- public cached full-kernel CLI output and metadata;
- existing full repository regression and contract suites.

The finite-q band vertex storage convention remains exactly

\[
(U_+^\dagger V U_-)^T,
\]

including the imaginary amplitude/phase cross channels.

## Runtime metadata

Optimized point JSON/CSV rows now report:

```text
eigensystem_cache_enabled
cached_subgrid_count
cached_eigensystem_count
cache_build_wall_seconds
integration_wall_seconds
total_compute_wall_seconds
```

These fields permit direct comparison with historical
`integration_wall_seconds` results.

## A/B benchmark

Use a fresh output path for each route.

Reference pointwise route:

```bash
python -m validation.commands.ward.bond_metric_full_kernel \
  --nk 128 --mx 3 --my 2 --subgrid-average auto \
  --chunk-size 1024 --max-points 20000 \
  --output /tmp/dwave_reference_n128_m3_2.csv
```

Optimized cached route:

```bash
python -m validation ward bond-metric-full-kernel \
  --nk 128 --mx 3 --my 2 --subgrid-average auto \
  --chunk-size 1024 --max-points 20000 \
  --output /tmp/dwave_cached_n128_m3_2.csv
```

Compare the two JSON rows numerically and compare wall times.  The optimized
route remains diagnostic-only and does not establish a production reference.
