# finite-q deterministic panel-adaptive transverse contract

## Status

`deterministic_panel_adaptive` remains the active transverse controller candidate. It
is diagnostic until a common independent reference passes the real-`nk=1256`
acceptance suite and the complete representative-q scan.

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

The rejected SciPy GK21 backend, wrapper, CLI, tests, and dedicated document have
been removed from the active tree. Its rejection evidence remains in Git history and
the PR record. Fixed/composite Gauss is an offline independent reference, not a
runtime fallback.

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
periodic transverse coordinates.

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

A split-history correction is active: replacing a CC33 parent by two CC9 children
retains the parent error envelope, the direct parent-versus-children discrepancy,
and a floor from the raw child estimates. Parent-observed point scales are inherited.
This removes the artificial error reset seen in the earlier split implementation.

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
candidate with the largest deterministic error-benefit score is executed.

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
is not converted into sigma, reflection, or logdet.

## Refinement trace

Every completed operation is serialized with:

```text
step
stage = primary | audit
selected panel bounds
old CC order
operation = p_refine | split_history
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

## Independent composite-Gauss reference

The existing complete-orbit fixed-Gauss backend also supports an equal-panel
composite rule without introducing a parallel implementation:

```text
transverse_order = total transverse nodes
panel_count = 16
C192 = 16 x GL12
C224 = 16 x GL14
C256 = 16 x GL16
```

The reference integrates a complete interval of length `2*pi`, may compare multiple
periodic cuts, and uses the shared mixed absolute/relative physical gates. It is
currently being evaluated only at the two difficult diagonal points and Matsubara
indices `1,2`.

Acceptance requires both:

```text
C256-C224 full-sigma mixed ratio <= 1
fixed-cut versus smooth-cut C256 full-sigma mixed ratio <= 1
```

plus Ward, sheet, reflection, and logdet gates. Only after this common reference
passes may the panel estimator be recalibrated and the representative-q certification
begin.

Failure at a q point must not create a q-specific rule. The outcome is either
acceptance of the common contract or a separately scoped redesign of the common
transverse/BZ representation.
