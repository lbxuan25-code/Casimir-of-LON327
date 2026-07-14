# Validation

`validation/` contains reproducible checks for the current two-band finite-q response contract. It is not the full Casimir outer integrator.

## Public command surface

All commands use:

```bash
python -m validation <group> <command> [options]
```

The outer-integration intake surface is deliberately small.

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

The total-Matsubara complete-orbit routes retain the qualified commensurate reference. The arbitrary-q routes are blocking preflight and qualification commands for the new fixed periodic-BZ backend; their existence does not establish a production response reference. Historical positive-only aliases are no longer public commands.

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

These routes localize numerical behavior or preserve forensic reproducibility. They never establish a production response reference and must not become dependencies of the full outer-integration runtime.

## Total Matsubara microscopic batch

`matsubara-orbit-gauss-crosscheck`, `orbit-gauss-preflight` and `total-orbit-gauss-scan` evaluate exact `n=0` and positive Matsubara frequencies in one complete-orbit callback with shared eigensystems.

- `n=0` uses the exact thermodynamic divided difference and is postprocessed as density/stiffness, static sheet response and static reflection.
- `n>0` is postprocessed as conductivity sheet response and positive-frequency reflection.
- `n=0` is never constructed by `sigma=-K/xi`.
- Primitive blocks are integrated before phase-Hessian pullback, Schur reduction, sheet conversion, reflection or logdet.
- The final single-point observable is `lno327.casimir.lifshitz_integrand.passive_sheet_logdet` in the common lab LT tangential-electric basis.

## Retained commensurate reference

The qualified complete-orbit path uses one full-period equal-panel composite Gauss-Legendre rule with no even, C4, axis, diagonal or q-direction symmetry reduction. Child processes compute full q orbits; the parent performs original-node-order complex Kahan reduction.

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
```

This backend is not forced onto arbitrary q. It remains the authority at commensurate q and the regression reference for the periodic-BZ implementation.

## Arbitrary-q periodic BZ backend

`ArbitraryQPeriodicBZContract-v1` uses an exact real crystal momentum on a fixed shifted `N x N` periodic midpoint lattice. It does not round q, change its direction or magnitude, or interpolate primitive response.

The implementation contract is:

```text
shared quadrature-independent primitive kernel
readonly q-independent material cache
exact q-dependent streamed canonical reduction blocks
exact n=0 + positive Matsubara shared shifted eigensystems
Goldstone/HS counterterm added once after full linear accumulation
normal Peierls operator identity checked before integration
integrated RHS-aware Ward checked after full integration
q_lab + angle-batch persistent-fork tasks
ordered parent collection and pickle-safe compact payloads
primary shift (1/2,1/2)
audit shifts (1/4,3/4) and (3/4,1/4)
```

The two blocking commands have different purposes:

- `arbitrary-q-performance-preflight` verifies readonly cache reuse, Matsubara eigensystem sharing, counterterm count, chunk invariance, process determinism, single-thread BLAS/OMP and real-hardware q-task speedup.
- `arbitrary-q-periodic-bz-qualification` compares commensurate q against complete-orbit, performs `N=256/384/512` arbitrary-q refinement and paired-shift audit, and checks a two-plate `0/17 degree` common-lab-LT logdet path.

Large-N qualification must not be run before the same-head performance preflight passes. CI runs only small deterministic contract tests; real speedup and large-N manifests are local blocking evidence.

## Exact-diagonal d-wave finding

The final pre-outer diagnostic established:

- response-level cut sensitivity is confined, at tested resolution, to exact `qx=qy` d-wave directions;
- the nearest tested off-diagonal direction, `(25,24)`, is `1.169139...` degrees from the diagonal and is strict at approximately `1e-6` response drift;
- exact diagonal `(6,6)`, `(12,12)` and `(24,24)` remain response-unresolved at low Matsubara frequency;
- every tested reflection and logdet cut drift remains below `1e-3`;
- Ward, exact-static Ward and the physical pipeline pass.

The arbitrary-q qualification preserves this policy: exact-diagonal primitive response may remain explicitly unresolved, but physical, reflection, logdet and grid-shift observable gates remain hard.

## Repository and output boundary

- `commands/`: CLI parsing and serialization only.
- `lib/`: tested validation algorithms and adapters.
- `outputs/`: local reproducible data; only compact summaries and metadata may be committed.
- `reports/`: current cross-validation summaries.
- `diagnostic` public routes: forensic tools, never runtime dependencies.

Raw CSV/JSON, figures, caches and arrays are ignored. Before cleaning local artifacts:

```bash
git clean -ndX validation/outputs outputs
git clean -fdX validation/outputs outputs
```

Do not run the destructive command while a job is active. Preserve needed compact evidence first as a `summary` or `status` artifact.

## Handoff

The implementation contracts are:

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
