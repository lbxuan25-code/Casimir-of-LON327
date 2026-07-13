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

Primary and audit snapshots are complete global primitive integrals. Each is
postprocessed exactly once and the resulting primitive groups, conductivity,
reflection, and logdet are compared.

## Common deterministic rule

The interval `[-pi, pi]` begins as four equal panels. Every panel uses nested
Clenshaw-Curtis rules:

```text
CC9 subset CC17 subset CC33
```

The initial complete state is CC17 on all four panels. Refinement is uniform in
algorithm, not in q direction:

1. compute physical-group errors from the nested high/low panel estimates;
2. select the panel with the largest normalized contribution to the worst control
   group;
3. upgrade `CC17 -> CC33` first;
4. if a CC33 panel remains dominant, bisect it and initialize both children at
   CC17;
5. repeat until the primary groupwise tolerance is satisfied;
6. save the primary snapshot;
7. continue from the same panel state until the tolerance multiplied by
   `audit_tolerance_factor` is satisfied;
8. save the tightened audit snapshot.

No direction label, q magnitude, or point name enters this decision tree.

## Groupwise error contract

For panel `p` and physical group `g`, the local error estimate is based on the
nested-rule difference:

```text
CC17 panel:  ||I17 - I9||
CC33 panel:  max(||I33 - I17||, 0.25 ||I17 - I9||)
```

A fixed safety factor of two is applied. The global group error is the conservative
sum over active panels.

Group scales are recomputed from all values in the current complete panel state.
This is safe in the explicit controller because all panel estimates and errors are
retained and all normalized ratios are recalculated whenever a larger amplitude is
discovered. The controller does not silently rescale an opaque external error.

For control group `g`:

```text
T_g = epsabs + epsrel * ||I_g|| / scale_g
R_g = (sum_p E_p,g / scale_g) / T_g
```

The primary or audit snapshot passes only when every control-group ratio is at most
one. Ward RHS is sampled on every node but has zero refinement weight; final
RHS-aware Ward validation remains authoritative.

## Budget and failure semantics

The primary and tightened audit share one hard cap on unique transverse
coordinates. Cache hits do not consume the cap. Complete microscopic orbit points
are counted separately.

Before any complete operation, the controller computes the exact number of missing
nodes for:

- one `CC17 -> CC33` panel upgrade; or
- both CC17 child panels created by a split.

The operation begins only when:

```text
current_unique + required_new <= hard_cap
```

If the operation would exceed the cap, it is not started. The result retains the
last complete global estimate, finite group errors and ratios, panel partition,
worst group, and worst panel. It fails closed with a structured panel-boundary
budget reason.

This differs from the rejected GK21 driver, which could exhaust the budget midway
through an opaque SciPy subdivision and lose the last complete global estimate.

## Primary/audit semantics

The tightened audit is not a second independent adaptive run. It is a strict
continuation of the primary panel state:

```text
initial state -> primary snapshot -> more refinement -> audit snapshot
```

All nodes, panel estimates, errors, and cache entries are reused. The two complete
snapshots are then compared using the original physical-group, conductivity,
reflection, and logdet gates.

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
- unique transverse evaluations, cache hits, and microscopic point evaluations;
- primary and audit group errors, tolerances, ratios, and dynamic scales;
- primary and audit panel counts, maximum depth, and refinement steps;
- worst physical group, worst panel bounds, panel order, and local ratio;
- primary/audit primitive-group differences;
- Matsubara-resolved Ward, conductivity, reflection, and logdet gates;
- geometry, evaluator, quadrature, and total wall times;
- structured failure reason;
- diagnostic/readiness flags.

## Correctness tests

The controller is tested independently of the d-wave model for:

- CC9/17/33 nesting and normalization;
- constants and polynomial moments;
- complex-vector BZ averages;
- a localized periodic analytic integrand;
- shared primary/audit panel state;
- zero-weight monitor groups;
- exact panel-boundary budget stopping with a finite retained snapshot.

A reduced real d-wave CLI smoke verifies complete-orbit construction, batched
microscopic evaluation, primitive unpacking, global postprocessing, and result
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
