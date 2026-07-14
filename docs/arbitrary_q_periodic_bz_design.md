# Arbitrary-q periodic BZ implementation decision

Current hard state:

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

## Frozen architecture

- Exact `q_crystal = R(-theta) q_lab`; no q/angle rounding, wrapping, nearest-commensurate replacement or primitive interpolation.
- Fixed shifted even-`N`, `N x N` full periodic midpoint BZ lattice.
- Primary shift `(1/2,1/2)`; formal audit shifts `(1/4,3/4)` and `(3/4,1/4)`.
- One physical batched q-workspace implementation. The operator-aware entry point is a thin wrapper over the established builder.
- Operator diagnostics reuse the same shifted Hamiltonians and Peierls vertices as the response.
- Exact zero and all positive Matsubara frequencies share each q workspace.
- Full linear primitive accumulation precedes phase-Hessian pullback, Schur, sheet, reflection and logdet processing.
- The Goldstone/HS counterterm is added exactly once after full-BZ accumulation.
- Persistent POSIX-fork parallelism uses `q_lab + angle_batch` tasks and ordered parent collection.
- A supplied material cache is used directly; no second `N^2` grid is built before response-cache lookup.

## Real runtime batching

The numerical and compute batch sizes now have different jobs:

```text
runtime chunk:
  build shifted Hamiltonians, eigensystems, vertices and Kubo factors once

canonical block:
  slice linear ingredients and define deterministic Kahan reduction boundaries
```

For a nonzero q, one runtime chunk performs two batched `np.linalg.eigh` calls, regardless of how many canonical blocks it contains. For example at fixed canonical size 4096:

```text
runtime 4096  -> one q workspace per 4096 points
runtime 16384 -> one q workspace per 16384 points
```

The packed result must remain invariant while q-workspace/eigensystem call counts change. `runtime_chunk_size` therefore controls actual throughput and peak q-dependent memory; it is still excluded from the response-cache numerical identity because canonical blocks define the floating-point contract.

## Paired shift contract

The paired estimate is formed only at the linear packed-primitive level:

```text
paired_packed = 0.5 * (packed_A + packed_B)
```

Then exactly one phase-Hessian, Schur, sheet, reflection and logdet pipeline is applied. Nonlinear averages of `R_A/R_B` or `logdet_A/logdet_B` are diagnostic spreads, not quadrature references.

`paired_average_arbitrary_q_results` requires:

```text
same material_state_fingerprint
same N, BZ convention and point count
inversion-related audit grids
formal shifts exactly (1/4,3/4) and (3/4,1/4)
same q, Matsubara list, phase policy, block policy and Ward tolerances
```

A dedicated `PairedShiftProfile-v1` reports summed source point evaluations, q-workspace builds, eigensystem calls, block counts and stage timings. The two source counterterms average to one effective counterterm.

## Complete formal policy

Only `ArbitraryQFormalPolicyV2` may establish formal evidence. More stringent settings are allowed where the policy specifies upper/lower bounds; looser or different physical settings are diagnostic-only.

### Performance workload

`ArbitraryQPerformanceWorkloadV2` freezes:

```text
model workload: symmetry_bdg_2band_bond_endpoint_gauge_v1
pairings: spm,dwave
N >= 128
Matsubara includes 0,1,2,4,8
canonical block = 4096
runtime chunks include 4096,16384
comparison atol <= 2e-12
comparison rtol <= 2e-11
T = 10 K
delta0 = 0.1 eV
eta = 1e-8 eV
outer workload: >=8 tasks, >=4 workers
qualification-primary workload: 4 tasks, 4 workers
qualification-audit workload: 1 task, 1 worker
outer speedup >= 4
outer CPU/wall >= 4
pool startup+shutdown overhead <= 0.05
```

The manifest must contain passed records for all three workload classes:

```text
outer_q_batch_v2
qualification_primary_v2
qualification_audit_v2
```

### Numerical matrix

`ArbitraryQQualificationMatrixV2` freezes:

```text
pairings: spm,dwave
N includes 256,384,512
reference nk = 1256
reference order >= 384
reference panel count = 16
reference workers = 8
reference task size = 4
Matsubara includes 0,1,8
primitive rtol <= 1e-3, atol <= 1e-12
reflection rtol <= 3e-4, atol <= 1e-12
logdet rtol <= 3e-4, atol <= 1e-14
diagonal observable rtol <= 1e-3, atol <= 1e-12
Ward tolerance <= 1e-7
Ward absolute tolerance <= 1e-12
T = 10 K
delta0 = 0.1 eV
eta = 1e-8 eV
separation = 20 nm
canonical block = 4096
runtime chunk = 16384
primary workers = 4
audit workers = 1
```

## Clean-source evidence

Formal evidence is not identified by `HEAD` alone. Before performance, before qualification and after qualification, the public commands require a clean worktree and record:

```text
git_head
git_tree_sha
tracked_index_fingerprint
source_tree_fingerprint
worktree_clean = True
```

The source fingerprint combines the commit, tree object and tracked index. Any tracked or untracked worktree change makes the run nonformal. The public gate requires identical source provenance across the performance manifest, current checkout, numerical core output and post-run checkout.

The numerical core can emit only:

```text
diagnostic_result_passed
diagnostic_result_failed
```

Only the clean-source public gate may promote it to:

```text
qualified_for_diagnostic_outer_integration
```

## Performance evidence

The formal preflight records and gates:

```text
actual child and parent BLAS threadpool counts
actual q-workspace and batched-eigh call counts
short/full Matsubara eigensystem equality
runtime-chunk packed-result equality
serial/process equality
cache-on/off equality and timing
worker RSS/PSS
serialized IPC payload bytes
parent collection overhead
pool startup and shutdown after close/join
three workload classes and their actual worker utilization
```

BLAS/OpenMP variables must be set before Python imports NumPy. Child workers verify the real runtime through `threadpoolctl`; environment strings alone are insufficient.

## Final two-plate qualification

Formal microscopic qualification gates the quantity consumed by a future Casimir outer integrator, not only isolated plate responses. For each pairing and Matsubara index it directly constructs:

```text
plate 1: theta = 0 degrees
plate 2: theta = 17 degrees
common lab LT tangential-electric basis
logdet(I - R1 R2 exp(-2 kappa d))
```

It requires:

```text
two-plate logdet at N=256,384,512
N256 -> N384 and N384 -> N512 convergence
audit-A two-plate logdet
audit-B two-plate logdet
primitive-paired plate 1
primitive-paired plate 2
paired two-plate logdet
A vs B two-plate sensitivity
primary N512 vs paired two-plate sensitivity
all plate/operator/Ward/static/sheet/reflection/passive gates at every source result
```

Single-plate convergence is retained but cannot substitute for this nonlinear final-observable gate.

## Momentum support versus qualification

The implementation syntactically supports the principal domain:

```text
|q_x| <= pi
|q_y| <= pi
```

and rejects values outside it without wrapping. This is **not** a numerically qualified outer-integration envelope.

The current formal matrix covers discrete axis, generic, near-diagonal, exact-diagonal and 17-degree-rotated vectors, and reports their maximum tested component/norm. It explicitly records:

```text
qualified_outer_q_envelope_established = False
continuous_angle_coverage_established = False
outer_tail_requirement_bound = False
```

After an outer separation range and tail tolerance are chosen, a separate envelope qualification must cover the required radii and angles up to the resulting `q_max`. A future outer builder must reject nodes outside that manifest envelope.

## Qualification order

1. Complete-orbit regression and renewed target-machine timing.
2. Tiny unified-builder, runtime-batch, cache, paired-shift, Ward and two-plate tests.
3. Clean-head formal performance preflight covering all three workload classes.
4. Without modifying source, run the clean-source public numerical gate at `N=256/384/512`.
5. Inspect the discrete q-coverage record; do not claim an outer q envelope.
6. Only after later outer q/angle/Matsubara/tail convergence may a full energy calculation begin.

The current branch remains diagnostic-only until the target-machine manifests are produced.
