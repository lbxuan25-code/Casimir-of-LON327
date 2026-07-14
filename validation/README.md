# Validation

`validation/` contains reproducible checks for the current two-band finite-q response contract. It is not the full Casimir outer integrator.

## Public command surface

All commands use:

```bash
python -m validation <group> <command> [options]
```

### Main contract commands

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

The complete-orbit routes retain the qualified commensurate reference. The arbitrary-q routes implement a blocking formal preflight and a same-head formal qualification gate for the fixed periodic-BZ backend. Their existence does not establish a production reference.

### Diagnostic-only commands

```text
diagnostic dwave-small-xi
diagnostic bond-metric-positive
diagnostic dwave-orbit-adaptive
diagnostic dwave-orbit-panel-adaptive
diagnostic dwave-orbit-evaluator-profile
diagnostic dwave-orbit-integrand-profile
diagnostic dwave-diagonal-width-scan
diagnostic dwave-orbit-gauss-crosscheck
diagnostic dwave-orbit-certification-scan
```

These routes localize numerical behavior or preserve forensic reproducibility. They never authorize production input.

## Retained complete-orbit reference

The qualified commensurate backend uses one full-period equal-panel composite Gauss-Legendre rule with no even, C4, axis, diagonal or q-direction symmetry reduction:

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
```

Exact `n=0` and positive Matsubara frequencies share eigensystems. Primitive blocks are integrated before phase-Hessian pullback, Schur reduction, sheet conversion, reflection or logdet. Zero frequency is never constructed from `sigma=-K/xi`.

The complete-orbit evaluator and the arbitrary-q backend now use the same quadrature-independent primitive kernel. The stable complete-orbit profiling contract remains unchanged, while its operator identity audit reuses q-workspace Hamiltonians and vertices instead of repeating them.

## Arbitrary-q periodic BZ backend

`ArbitraryQPeriodicBZContract-v2` provides:

```text
exact q_crystal = R(-theta) q_lab
no rounding, wrapping, nearest-commensurate replacement or interpolation
fixed shifted N x N full periodic midpoint lattice
readonly q-independent MaterialGridCache-v2
streamed canonical reduction blocks with ordered complex Kahan accumulation
exact n=0 + positive Matsubara shared q workspace
Goldstone/HS counterterm added exactly once
q_lab + angle-batch persistent POSIX-fork tasks
ordered streaming result collection
CrystalResponseCache-v2 with complete numerical-policy identity
```

When a material cache is supplied, the entry point uses `material_cache.grid`; it does not rebuild or traverse a second `N^2` grid before a response-cache lookup.

### Ward layers

- The machine-level pointwise gate is the normal Peierls operator identity only.
- The operator audit is computed from Hamiltonians and vertices already constructed by the q workspace.
- The integrated response uses the existing RHS-aware Ward validation.
- Exact zero additionally keeps the strict-static longitudinal gate fail-closed.

Tiny unit grids prove operator and integrated algebraic closure and the positive-frequency sheet/reflection/logdet path. They are not allowed to impersonate converged zero-mode phase or longitudinal physics. Those gates belong to formal `N=256/384/512` qualification.

### Paired shift audit

Audit shifts are:

```text
A = (1/4,3/4)
B = (3/4,1/4)
```

The paired estimate is defined only at the linear primitive level:

```text
paired_packed = 0.5 * (packed_A + packed_B)
```

Exactly one phase-Hessian, Schur, sheet, reflection and logdet pipeline is applied to `paired_packed`. `A/B` reflection and logdet differences are reported as independent spreads; their nonlinear averages are not quadrature references.

Every complete-orbit reference, primary `N`, audit shift and paired-primitive result must independently pass the relevant operator, integrated Ward, strict-static, sheet, reflection and passive-logdet gates.

## Frozen formal policy

Only `ArbitraryQFormalPolicyV1` can establish a formal pass. CLI values may be stricter but cannot be looser.

Performance policy freezes at least:

```text
pairings include spm,dwave
N >= 128
q tasks >= 8
workers >= 4
Matsubara includes 0,1,2,4,8
canonical block = 4096
runtime chunks include 4096,16384
minimum speedup >= 4
minimum CPU/wall >= 4
pool overhead <= 0.05
```

Numerical policy freezes at least:

```text
pairings include spm,dwave
N values include 256,384,512
reference nk = 1256
reference order >= 384
Matsubara includes 0,1,8
primitive rtol <= 1e-3
reflection/logdet rtol <= 3e-4
diagonal observable rtol <= 1e-3
```

A formal performance manifest contains:

```text
formal policy id and pass state
config fingerprint
exact command
git head
hardware fingerprint
execution/worker/thread policy
actual BLAS threadpool report
actual eigensystem call counters
RSS/PSS and IPC payload metrics
cache-on/off comparison
parent collection overhead
```

The public qualification route rejects missing, stale, forged, nonformal or execution-incompatible manifests before any large calculation begins.

The underlying numerical core can only emit:

```text
diagnostic_result_passed
diagnostic_result_failed
```

Only the public same-head gate may promote a passed result to:

```text
qualified_for_diagnostic_outer_integration
```

## Qualification execution path

The large-N qualification uses the same cached persistent q-task backend measured by the performance preflight.

For each pairing it builds only:

```text
primary N=256 cache
primary N=384 cache
primary N=512 cache
audit-A N=512 cache
audit-B N=512 cache
```

Each primary cache evaluates axis, generic, near-diagonal, exact-diagonal and rotated-q cases through one shared q-task batch. The two-plate `0/17 degree` common-lab-LT path reuses the same final primary context.

## Microscopic q domain

The current backend explicitly rejects:

```text
|q_x| > pi
|q_y| > pi
```

It never silently wraps q. Before a production outer integral is allowed, the outer `Q` cutoff and tail convergence must show that all rotated crystal momenta remain within the validated domain and that the neglected tail is small before a BZ boundary is reached.

## Exact-diagonal d-wave policy

The retained reference established that response-level cut sensitivity is confined, at tested resolution, to exact `qx=qy` d-wave directions, while nearby off-diagonal directions are strict and reflection/logdet sensitivity remains below `1e-3`.

The arbitrary-q qualification preserves that policy: exact-diagonal primitive response may remain explicitly unresolved, but operator, integrated Ward, strict-static, reflection, logdet and shift-sensitivity gates remain hard.

## Output and readiness boundary

Raw CSV/JSON, figures, caches and arrays remain local and ignored. Only compact summaries and metadata may be committed.

Authoritative documents:

```text
docs/full_outer_integration_handoff.md
docs/arbitrary_q_periodic_bz_design.md
scripts/casimir/README.md
```

Current hard state:

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```
