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
```

Only the total-Matsubara routes may be used to qualify input for the future full outer integral. Historical positive-only aliases are no longer public commands.

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

## Retained transverse method

The qualified main path uses one full-period equal-panel composite Gauss-Legendre rule with no even, C4, axis, diagonal or q-direction symmetry reduction. Child processes compute full q orbits; the parent performs original-node-order complex Kahan reduction.

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
```

Multi-process runs must pin BLAS/OpenMP thread counts to one. A real-`nk` preflight manifest from the same Git head is mandatory before any formal scanner run.

## Exact-diagonal d-wave finding

The final pre-outer diagnostic established:

- response-level cut sensitivity is confined, at tested resolution, to exact `qx=qy` d-wave directions;
- the nearest tested off-diagonal direction, `(25,24)`, is `1.169139...` degrees from the diagonal and is strict at approximately `1e-6` response drift;
- exact diagonal `(6,6)`, `(12,12)` and `(24,24)` remain response-unresolved at low Matsubara frequency;
- every tested reflection and logdet cut drift remains below `1e-3`;
- Ward, exact-static Ward and the physical pipeline pass.

This permits a diagnostic full outer-integration trial with explicit diagonal sensitivity variants. It does not establish a production response reference.

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

The next implementation window must begin with:

```text
docs/full_outer_integration_handoff.md
scripts/casimir/README.md
```

Current hard state:

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```
