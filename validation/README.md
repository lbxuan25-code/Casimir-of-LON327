# Validation

`validation/` contains reproducible checks for the current two-band finite-q response contract. It is not the full Casimir outer integrator.

## Public command surface

```bash
python -m validation <group> <command> [options]
```

Main retained and blocking commands:

```text
ward commensurate
ward bond-metric-full-kernel
ward bond-metric-family

static nk-scan
static dwave
static dwave-orbit
static projection-scan
static quadrature-compare

matsubara positive-point
matsubara total-orbit-timing-profile
matsubara matsubara-orbit-gauss-crosscheck
matsubara orbit-gauss-preflight
matsubara total-orbit-gauss-scan
matsubara arbitrary-q-performance-preflight
matsubara arbitrary-q-periodic-bz-qualification
```

The complete-orbit routes retain the qualified commensurate reference. The arbitrary-q routes implement a target-machine performance preflight and a clean-source public qualification gate. Their existence does not establish production readiness.

Diagnostic routes remain forensic tools and cannot authorize outer input.

## Retained complete-orbit reference

The commensurate backend uses full-period equal-panel composite Gauss-Legendre integration without even, C4, axis, diagonal or q-direction symmetry reduction:

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
```

Exact `n=0` and positive Matsubara values share eigensystems. Primitive blocks are integrated before phase-Hessian pullback, Schur reduction, sheet conversion, reflection or logdet. Zero frequency is never constructed from `sigma=-K/xi`.

The retained reference and arbitrary-q backend share the same primitive kernel and the same single q-workspace implementation. The operator-enabled entry point is a thin wrapper, so no copied shifted-Hamiltonian/vertex implementation remains.

## Arbitrary-q periodic BZ backend

`ArbitraryQPeriodicBZContract-v3` provides:

```text
exact q_crystal = R(-theta) q_lab
no q rounding, wrapping, nearest-grid substitution or interpolation
fixed shifted even-N, N x N full periodic midpoint lattice
readonly MaterialGridCache-v3
material-state fingerprint separate from grid fingerprint
CrystalResponseCache-v3 with complete numerical-policy identity
one established q-workspace implementation with optional operator diagnostics
runtime-sized q workspace and eigensystem batches
canonical-block deterministic Kahan reduction
exact zero + positive Matsubara shared q workspace
Goldstone/HS counterterm added once
q_lab + angle-batch persistent POSIX-fork tasks
ordered streaming result collection
actual child BLAS threadpool verification
```

When a material cache is supplied, the entry point uses `material_cache.grid`; it does not build a second `N^2` grid before response-cache lookup.

### Runtime versus canonical blocks

```text
runtime_chunk_size:
  actual shifted Hamiltonian/eigensystem/vertex/Kubo batch

canonical_reduction_block_size:
  fixed linear contribution and Kahan addition boundary
```

At nonzero q each runtime chunk performs two batched eigensystem calls. Changing runtime 4096 to 16384 changes q-workspace/eigensystem call counts and memory width, while canonical boundaries and packed output remain invariant.

### Ward layers

- Machine-scale pointwise checks apply only to the normal Peierls operator identity.
- Operator diagnostics reuse q-workspace Hamiltonians and Peierls vertices.
- Integrated response uses RHS-aware Ward validation.
- Exact zero retains the strict-static longitudinal gate.
- Tiny grids prove algebra and positive-frequency plumbing, not converged zero-mode physics.

## Paired-shift audit

Formal audit shifts are:

```text
A = (1/4,3/4)
B = (3/4,1/4)
```

The primary paired estimate is:

```text
paired_packed = 0.5 * (packed_A + packed_B)
```

One nonlinear postprocessing pipeline follows. `A/B` reflection and logdet differences remain spread diagnostics only.

Pairing is rejected unless material state, N, BZ convention, point count, q/frequency/policy identity and inversion-related formal shifts match. `PairedShiftProfile-v1` sums both source evaluations and reports one effective counterterm after averaging.

## Frozen formal policy

Only `ArbitraryQFormalPolicyV2` can establish formal evidence.

### Performance contract

`ArbitraryQPerformanceWorkloadV2` includes:

```text
pairings spm,dwave
N >= 128
Matsubara includes 0,1,2,4,8
canonical block = 4096
runtime chunks include 4096,16384
comparison atol <= 2e-12
comparison rtol <= 2e-11
T = 10 K, delta0 = 0.1 eV, eta = 1e-8 eV
outer_q_batch_v2: >=8 tasks, >=4 workers
qualification_primary_v2: 4 tasks, 4 workers
qualification_audit_v2: 1 task, 1 worker
outer speedup >= 4
outer CPU/wall >= 4
pool startup+shutdown overhead <= 0.05
```

### Numerical contract

`ArbitraryQQualificationMatrixV2` includes:

```text
pairings spm,dwave
N includes 256,384,512
reference nk = 1256
reference order >= 384
reference panel count = 16
reference workers/task size = 8/4
Matsubara includes 0,1,8
primitive rtol/atol <= 1e-3/1e-12
reflection rtol/atol <= 3e-4/1e-12
logdet rtol/atol <= 3e-4/1e-14
diagonal observable rtol/atol <= 1e-3/1e-12
Ward tolerance/absolute tolerance <= 1e-7/1e-12
T = 10 K, delta0 = 0.1 eV, eta = 1e-8 eV
separation = 20 nm
canonical/runtime = 4096/16384
primary/audit workers = 4/1
```

All quantities that affect workload, physical point or pass/fail are part of the formal config fingerprint.

## Clean-source gate

`git rev-parse HEAD` alone is insufficient. Formal commands require:

```bash
git status --porcelain --untracked-files=all
```

to be empty and bind evidence to:

```text
git_head
git_tree_sha
tracked_index_fingerprint
source_tree_fingerprint
worktree_clean = True
```

The public gate compares the performance manifest, current source tree, numerical-core output and post-run source tree. Dirty or changed trees can run only with `--diagnostic-nonformal` and cannot authorize anything.

The numerical core emits only:

```text
diagnostic_result_passed
diagnostic_result_failed
```

Only the public gate can promote it to:

```text
qualified_for_diagnostic_outer_integration
```

## Performance evidence

The formal preflight measures:

```text
outer, qualification-primary and qualification-audit workloads
serial/process equality
runtime 4096/16384 equality
real q-workspace and batched-eigh call counts
short/full Matsubara eigensystem equality
cache-on/off equality and timing
actual parent and child BLAS threadpool counts
worker RSS/PSS
IPC payload bytes
parent collection overhead
pool startup and shutdown measured after close/join
```

BLAS/OpenMP variables must be exported before Python starts.

## Large-N qualification

For each pairing, qualification builds only:

```text
primary N=256 cache
primary N=384 cache
primary N=512 cache
audit-A N=512 cache
audit-B N=512 cache
```

Primary contexts use four workers for four q tasks. Audit contexts use one worker for one task, matching measured workloads rather than opening eight idle workers.

Every complete-orbit reference, every primary N, both audit shifts and paired results must independently pass their operator, integrated Ward, strict-static, sheet, reflection and passive-logdet gates.

### Final consumed two-plate observable

The common-lab result consumed by future Casimir integration is directly gated:

```text
plate 1 theta = 0 degrees
plate 2 theta = 17 degrees
logdet(I - R1 R2 exp(-2 kappa d))
```

For each Matsubara index qualification requires:

```text
N=256/384/512 two-plate logdet
N refinement
A and B two-plate logdet
primitive-paired plate 1 and plate 2
paired two-plate logdet
A vs B sensitivity
primary N512 vs paired sensitivity
all source and paired plate physical gates
```

Single-plate convergence cannot substitute for this nonlinear final-observable gate.

## Momentum support versus qualification

The implementation rejects components outside:

```text
|q_x| <= pi
|q_y| <= pi
```

This is the syntactically supported principal domain, not a qualified outer envelope. The current matrix reports only its discrete tested q vectors and maximum tested norm/component, with:

```text
qualified_outer_q_envelope_established = False
continuous_angle_coverage_established = False
outer_tail_requirement_bound = False
```

A later outer configuration must determine the q range required by separation, angle range and tail tolerance, then establish a separate numerical envelope manifest. An outer builder must reject nodes beyond that envelope.

## Current state

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

Authoritative handoff documents:

```text
docs/full_outer_integration_handoff.md
docs/arbitrary_q_periodic_bz_design.md
scripts/casimir/README.md
```
