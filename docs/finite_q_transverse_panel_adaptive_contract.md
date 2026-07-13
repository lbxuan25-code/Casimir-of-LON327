# finite-q transverse quadrature contract

## Status

The positive-Matsubara finite-q response still has no accepted production
transverse reference.

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

The active production candidate remains the deterministic nested panel controller.
The fixed/composite Gauss-Legendre backend is an independent offline reference used
to determine whether panel error estimates reflect real integration error. Neither
backend is a runtime fallback for the other.

Rejected GK21 and earlier panel behavior remain reproducible through git history,
validation outputs, and this document. New version-suffixed quadrature modules must
not be added for each experiment. Common behavior belongs in the existing panel or
Gauss backend.

## Physical invariants

At every transverse coordinate `t`, the callback evaluates one complete exact
commensurate q orbit, including complementary origins when required. It returns one
packed primitive vector containing electromagnetic, collective, mixed, and Ward-RHS
blocks for the full Matsubara batch.

A transverse integrator may change only nodes and weights. It must not perform any of
these operations locally:

- bond-metric correction;
- amplitude/phase Schur complement;
- collective projection;
- sheet construction;
- reflection construction;
- passive logdet evaluation.

Those operations are applied once to a complete global primitive integral.

## Full-period and symmetry contract

Every candidate integrates an interval of exact length `2*pi`.

```text
full_transverse_period_integrated = True
symmetry_reduction_applied = False
q_direction_special_case = False
```

A periodic cut may move from `t0` to another `t0`, because
`f(t + 2*pi) = f(t)`. The implementation must not assume:

- `f(t) = f(-t)`;
- C4 equivalence between q directions;
- axis/diagonal equivalence;
- any relation that fails after explicit C4 breaking.

This is required for future anisotropic response and Casimir torque calculations.

## Active nested panel controller

The controller uses eight initial CC9 panels and the nested sequence

```text
CC5 -> CC9 -> CC17 -> CC33
```

where CC9, CC17, and CC33 are active panel states and CC5 is an embedded estimator.
Candidate operations are selected under one hard unique-node budget:

```text
CC9  -> CC17
CC17 -> CC33
CC33 -> two CC9 children
```

The split-history correction prevents a CC33 parent from being replaced by two raw
CC9 error estimates that forget the parent's high-order information. The combined
child envelope retains:

- the parent group error;
- the direct parent-versus-children integral discrepancy;
- a floor from the two raw child CC9-CC5 estimates;
- all observed parent point scales.

The correction removed the large artificial error-ratio jumps seen in the earlier
split behavior. It did not make the real 256-node acceptance suite pass.

### Real `nk=1256` split-history results

```text
reference (6,4):
  primary passed at 204 nodes
  tightened audit stopped at ratio 1.990

diagonal_mid (6,6):
  primary stopped at ratio 25.04

diagonal_min (1,1):
  primary stopped at ratio 59.83
```

Ward and physical postprocessing remained healthy wherever a complete primitive
snapshot was available. The remaining blocker is transverse integration.

## Shared primary/audit semantics

The audit is a strict continuation of the primary panel state:

```text
initial state -> primary snapshot -> additional refinement -> audit snapshot
```

A failed audit remains a quadrature diagnostic. It is not converted into sigma,
reflection, or logdet, and it cannot generate a misleading zero primary/audit
difference.

## Fixed and composite Gauss reference backend

`integrate_commensurate_orbit_gauss_aggregate` now supports both global and equal-
panel Gauss-Legendre rules through one implementation.

```text
transverse_order = total number of transverse nodes
panel_count       = number of equal panels
panel_order       = transverse_order / panel_count
```

`transverse_order` must be divisible by `panel_count`.

Examples:

```text
G224: panel_count=1,  panel_order=224
C224: panel_count=16, panel_order=14
C256: panel_count=16, panel_order=16
```

The total microscopic cost remains

```text
transverse_order * nk * number_of_orbit_origins
```

so C224 and G224 have the same transverse node and complete-orbit point counts.
Adjacent panels use interior Gauss nodes and therefore do not duplicate boundaries.
Complex Kahan summation is retained across all nodes.

### Periodic cuts

The backend accepts an explicit `integration_start` and integrates

```text
[integration_start, integration_start + 2*pi]
```

The public crosscheck command accepts repeated `--integration-start` arguments. This
allows the same composite rule to be checked at a fixed cut and at a smooth cut
without adding a second backend or assuming a spatial symmetry.

## Independent global-Gauss evidence

Global G192, G224, and G256 all passed Ward and physical construction, but the full
sigma tensor did not enter a stable monotone sequence at the difficult diagonal
points.

The largest observed consecutive differences included:

```text
diagonal_mid n=1:
  G224-G192 sigma relative = 6.77e-3
  G256-G224 sigma relative = 1.45e-2

diagonal_min n=1:
  G224-G192 sigma relative = 7.02e-3
  G256-G224 sigma relative = 3.62e-3
```

Reflection and the 20 nm, zero-angle logdet were already stable at approximately
`1e-4` to `1e-5`. That does not establish a production response reference because
the remaining sigma error is concentrated in the longitudinal-transverse splitting
channel and may matter after rotation or explicit C4 breaking.

A single global G288 run is therefore not the next acceptance step.

## Comparison gates

Matrix comparisons use the shared mixed gate

```text
||A-B|| <= atol + rtol * max(||A||, ||B||)
```

rather than a pure relative error for every tensor channel. Near-zero C4 residuals
and antisymmetric components remain diagnostics, not independent pure-relative hard
gates.

The command preserves raw absolute and relative differences and additionally writes
the normalized mixed-gate ratio. It records three comparison families:

- current rule versus an external reference CSV;
- one total order versus the previous total order at the same cut;
- one periodic cut versus the baseline cut at the same total order.

## Composite reference experiment

The next fixed experiment uses the existing Gauss command with

```text
panel_count = 16
total orders = 192, 224, 256
panel orders = 12, 14, 16
Matsubara indices = 1, 2 initially
```

Run both difficult points:

```text
diagonal_mid (6,6)
diagonal_min (1,1)
```

and two explicit cuts:

```text
fixed cut  = -pi
smooth cut = the already serialized panel-controller integration_start
```

The experiment passes only if:

```text
C256-C224 full-sigma mixed ratio <= 1
AND fixed-cut versus smooth-cut C256 full-sigma mixed ratio <= 1
AND reflection and logdet mixed ratios <= 1
AND Ward and the complete physical pipeline pass
```

If C224/C256 pass, composite Gauss becomes the independent estimator-calibration
reference. If they fail, the common representation needs more than 256 nodes or a
new generic variable transformation; no q-specific rule or weakened safety factor
is allowed.

## Maintenance policy

- Do not add another `v4`, `composite_v1`, or q-specific backend.
- Extend the existing panel or Gauss implementation when behavior is genuinely
  common.
- Keep rejected numerical outputs and concise documented conclusions, not multiple
  active runtime paths.
- Consolidate temporary split-history code into the main panel module after the
  composite reference direction is decided.
- Do not delete legacy scalar/reference material until an independent production
  reference passes.
- Never merge the draft PR while `valid_for_casimir_input` is false.
