# finite-q transverse adaptive GK21 contract

## Status

`adaptive_gk21` is the only transverse integration candidate being advanced toward
production.  It remains diagnostic until the decisive numerical acceptance suite
and the complete representative-q scan pass.

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

The historical periodic-nested and fixed-Gauss implementations remain available
only as offline references.  They are not runtime fallbacks for this candidate.

## Physical invariants

At every transverse coordinate `t`, the callback must evaluate the complete exact
commensurate q orbit, including required complementary origins.  It returns one
packed primitive vector containing electromagnetic, collective, mixed, and Ward
RHS blocks for the full Matsubara batch.

The transverse integrator may change only the `t` nodes and weights.  The following
operations are forbidden at a node or panel level:

- nearest-neighbour bond metric application;
- amplitude/phase Schur complement;
- collective projection;
- sheet construction;
- reflection construction;
- passive logdet evaluation.

Each complete global primitive integral is postprocessed once.  The primary and
tightened-audit estimates are postprocessed independently and then compared.

## Numerical contract

The candidate uses one fixed rule: SciPy adaptive GK21 on `[-pi, pi]`.

1. The 21 root-panel GK21 nodes are evaluated first and cached.
2. Physical-group scales are frozen from those samples.
3. The primary integral uses the requested final-BZ-average tolerances.
4. A tightened GK21 audit uses the same complete-orbit cache and tolerances scaled
   by `audit_tolerance_factor` (default `0.25`).
5. Ward RHS components share all nodes but do not independently drive refinement;
   the final Ward validation remains authoritative.
6. Scales are not enlarged or restarted during this first production-candidate
   implementation.  Observed-to-frozen scale ratios are diagnostic outputs.

For a final BZ average, `epsabs` is defined after division by `2*pi`.  The raw
`quad_vec` absolute tolerance and error estimate are therefore converted by the
same `2*pi` factor.

## Hard budget and failure semantics

The primary and audit passes share one hard cap on unique transverse evaluations,
defaulting to 256.  Cache hits do not consume the cap.  Complete microscopic orbit
points are counted separately.

A budget exception during a pass invalidates that pass.  No partially accumulated
SciPy integral is exposed as a trusted estimate.  If the audit exhausts the budget,
a completed primary estimate may be retained only as `diagnostic_only`.

Node-cap failure and wall-time failure are distinct:

- `unique_t > cap`: the integration representation/candidate failed the cost
  contract; optimizing the per-node evaluator does not fix the node count.
- `unique_t <= cap` but wall time is too high: profile and optimize the microscopic
  evaluator while preserving exact arithmetic and orbit coverage.

## Point gate

A point can pass only when all of the following are true:

```text
primary adaptive error passed
AND tightened audit passed
AND primitive physical-group agreement passed
AND sigma agreement passed
AND reflection agreement passed
AND logdet agreement passed
AND Ward passed
AND sheet validation passed
AND reflection constructed
AND passive logdet passed
AND unique transverse evaluations <= hard cap
```

No condition may be removed or weakened to force acceptance.

## Diagnostics that must be serialized

- strategy and SciPy version;
- unique transverse evaluations, cache hits, and microscopic point evaluations;
- primary/audit error estimates, tolerances, ratios, status, and subinterval count;
- frozen group scales and observed-to-frozen ratios;
- primary/audit primitive-group differences;
- primary/audit sigma, reflection, and logdet differences;
- worst reported intervals;
- geometry, evaluator, quadrature, and total wall times;
- structured failure reason;
- global diagnostic/readiness flags.

## Decisive acceptance suite

After correctness tests and low-risk evaluator profiling, run one fixed acceptance
suite, not a method sweep:

1. `reference (6,4)` adaptive GK21;
2. `diagonal_mid (6,6)` adaptive GK21 (difficult single-origin case);
3. `diagonal_min (1,1)` adaptive GK21 (difficult double-origin case);
4. `diagonal_mid` fixed Gauss G192/G224 offline comparison;
5. `diagonal_min` fixed Gauss G192/G224 offline comparison.

Failure at a q point must not create a q-specific integration rule.  The outcome is
either acceptance of this common contract or a separately scoped redesign of the
common transverse/BZ representation.
