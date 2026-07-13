# finite-q deterministic panel-adaptive transverse contract

## Status

`deterministic_panel_adaptive` is the only transverse integration candidate being
advanced toward production after rejection of the SciPy `quad_vec` GK21 driver.
It remains diagnostic until the fixed real-`nk=1256` acceptance suite and complete
representative-q scan pass.

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

Historical periodic, fixed-Gauss, and GK21 implementations are offline references
only. They are not runtime fallbacks.

## Physical invariants

At every transverse coordinate `t`, the callback evaluates the complete exact
commensurate q orbit, including complementary origins when required. It returns one
packed primitive vector containing electromagnetic, collective, mixed, and Ward-RHS
blocks for the full Matsubara batch.

The panel controller may alter only transverse nodes and weights. These operations
are forbidden at a point or panel level:

- nearest-neighbour bond metric application;
- amplitude/phase Schur complement;
- collective projection;
- sheet construction;
- reflection construction;
- passive logdet evaluation.

Primary and successful-audit snapshots are complete global primitive integrals. Each
is postprocessed exactly once and the resulting primitive groups, conductivity,
reflection, and logdet are compared. A failed audit snapshot remains a quadrature
diagnostic and is not postprocessed into electrodynamic observables.

## Full-period and symmetry contract

The controller always integrates one complete interval of length `2*pi`. It may move
the periodic cut, but it does not fold the domain or replace one side by a symmetry-
related copy.

```text
full_transverse_period_integrated = True
symmetry_reduction_applied = False
q_direction_special_case = False
```

In particular, the implementation does not assume:

- `f(t) = f(-t)`;
- C4 equivalence between q directions;
- axis/diagonal equivalence;
- any relation that would cease to hold after explicit C4 breaking.

This contract is required so that the same transverse integrator can later be used
for anisotropic response and Casimir torque calculations.

## Periodic cut and initial state

Sixteen equally spaced pilot coordinates are evaluated on the full period. These
pilot coordinates are exactly the boundary and midpoint coordinates reused by the
eight initial CC9 panels.

The cut is selected from those 16 candidates using the smallest local normalized
variation of the positive-weight physical control groups. Ward RHS and other
zero-weight monitor groups do not influence the cut.

After choosing `t0`, the full interval is

```text
[t0, t0 + 2*pi]
```

and begins as eight equal CC9 panels. The initial complete state contains 64 unique
periodic transverse coordinates, the same approximate initial cost as the earlier
four-panel CC17 prototype but with twice the spatial localization.

## Common deterministic rule

Every active panel uses one of the nested states

```text
CC9 subset CC17 subset CC33
```

CC5 is used only as the lower-order estimator for a CC9 panel. The allowed complete
operations are:

```text
CC9  -> CC17             approximately 8 new nodes
CC17 -> CC33             approximately 16 new nodes
CC33 -> two CC9 children approximately 14 new nodes
```

The exact cost is computed from the shared periodic cache before an operation begins.
No direction label, q magnitude, point name, or symmetry tag enters this decision
process.

## Groupwise error contract

For panel `p` and physical group `g`, the local error estimate is based on nested-rule
differences:

```text
CC9 panel:  ||I9  - I5||
CC17 panel: max(||I17 - I9||,  0.25 ||I9  - I5||)
CC33 panel: max(||I33 - I17||, 0.25 ||I17 - I9||)
```

A fixed safety factor of two is applied. The global group error is the conservative
sum over active panels.

Group scales are recomputed from all values in the current complete panel state.
This is safe in the explicit controller because all panel estimates and errors are
retained and every normalized ratio is recalculated when a larger amplitude is
discovered.

For control group `g`:

```text
T_g = factor * (epsabs + epsrel * ||I_g|| / scale_g)
R_g = weight_g * (sum_p E_p,g / scale_g) / T_g
```

The primary uses `factor = 1`; the tightened audit uses
`factor = audit_tolerance_factor`. A snapshot passes only when every positive-weight
control-group ratio is at most one. Ward RHS is sampled on every node but has zero
refinement weight; final RHS-aware Ward validation remains authoritative.

## Budget-aware refinement scheduling

At each step the controller enumerates the next complete operation for every active
panel. For each candidate it records:

- operation type and target order;
- exact number of missing unique nodes;
- normalized local physical-group error contribution;
- local contribution per required new node.

Candidates that do not fit the remaining hard budget are excluded. The feasible
candidate with the largest deterministic error-benefit score is executed. Thus a
16-node operation on the nominal worst panel cannot block a useful 8-node operation
elsewhere when only eight nodes remain.

The primary and tightened audit share one hard cap on unique transverse coordinates.
Cache hits do not consume the cap. Complete microscopic orbit points are counted
separately.

Before every p-refinement or split:

```text
current_unique + required_new <= hard_cap
```

If no complete operation fits, the controller retains the last complete global
estimate, finite group errors and ratios, panel partition, worst group, and worst
panel. It fails closed with a structured panel-boundary budget reason containing the
minimum additional complete-operation cost.

## Primary/audit semantics

The tightened audit is not an independent adaptive run. It is a strict continuation
of the primary panel state:

```text
initial state -> primary snapshot -> more refinement -> successful audit snapshot
```

All nodes, panels, errors, scales, and cache entries are reused. If the audit cannot
reach the tightened tolerance, its finite panel diagnostics are serialized, but it
is not converted into sigma, reflection, or logdet. Therefore a failed unchanged
audit can no longer produce misleading zero primary/audit differences.

## Refinement trace

Every completed operation is serialized with:

```text
step
stage = primary | audit
selected panel bounds
old CC order
operation = p_refine | split
required new unique nodes
unique nodes after operation
worst physical group before and after
global error ratio before and after
```

This trace is part of the numerical contract. It distinguishes genuine node-budget
failure from inefficient operation ordering or a conservative local error estimator.

## Point gate

A point can pass only when all of these are true:

```text
primary groupwise panel error passed
AND tightened groupwise panel audit passed
AND primary/audit primitive-group agreement passed
AND sigma agreement passed
AND reflection agreement passed
AND logdet agreement passed
AND primary and audit Ward passed
AND primary and audit sheet validation passed
AND reflection constructed
AND passive logdet passed
AND unique transverse evaluations <= hard cap
```

No gate may be weakened to force acceptance.

## Required diagnostics

The JSON result must serialize:

- strategy and nested quadrature rule;
- full-period cut, pilot count, initial panel count, and symmetry flags;
- unique transverse evaluations, cache hits, and microscopic point evaluations;
- primary and audit group errors, tolerances, ratios, and dynamic scales;
- primary and audit panel counts, maximum depth, and refinement steps;
- worst physical group, worst panel bounds, panel order, and local ratio;
- the complete refinement trace;
- primary/audit primitive-group differences when the audit succeeds;
- Matsubara-resolved Ward, conductivity, reflection, and logdet gates;
- geometry, evaluator, quadrature, and total wall times;
- structured failure reason;
- diagnostic/readiness flags.

## Correctness tests

The controller is tested independently of the d-wave model for:

- CC5/9/17/33 nesting and normalization;
- constants and polynomial moments;
- complex-vector BZ averages;
- a localized periodic analytic integrand;
- shared primary/audit panel state;
- a full-period asymmetric integrand with no even/C4 reduction;
- zero-weight monitor independence of cut and refinement sequence;
- exact panel-boundary budget stopping with a finite retained snapshot;
- use of a final feasible eight-node operation under a 72-node cap.

A reduced real d-wave CLI smoke verifies complete-orbit construction, batched
microscopic evaluation, primitive unpacking, global postprocessing, and v2 result
serialization.

## Decisive acceptance suite

Run one fixed real-`nk=1256` suite, not a method sweep:

1. `reference (6,4)` panel adaptive;
2. `diagonal_mid (6,6)` panel adaptive, difficult single-origin case;
3. `diagonal_min (1,1)` panel adaptive, difficult double-origin case;
4. existing or offline `diagonal_mid` fixed-Gauss G192/G224 comparison;
5. existing or offline `diagonal_min` fixed-Gauss G192/G224 comparison.

Failure at a q point must not create a q-specific integration rule. The outcome is
either acceptance of this common contract or a separately scoped redesign of the
common transverse/BZ representation.
