# finite-q transverse adaptive GK21 contract

## Status

`adaptive_gk21` is the only transverse integration candidate being advanced toward
production. It remains diagnostic until the decisive numerical acceptance suite
and the complete representative-q scan pass.

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

The historical periodic-nested and fixed-Gauss implementations remain available
only as offline references. They are not runtime fallbacks for this candidate.

## Physical invariants

At every transverse coordinate `t`, the callback must evaluate the complete exact
commensurate q orbit, including required complementary origins. It returns one
packed primitive vector containing electromagnetic, collective, mixed, and Ward
RHS blocks for the full Matsubara batch.

The transverse integrator may change only the `t` nodes and weights. The following
operations are forbidden at a node or panel level:

- nearest-neighbour bond metric application;
- amplitude/phase Schur complement;
- collective projection;
- sheet construction;
- reflection construction;
- passive logdet evaluation.

Each complete global primitive integral is postprocessed once. The primary and
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
   implementation. Observed-to-frozen scale ratios are diagnostic outputs.

For a final BZ average, `epsabs` is defined after division by `2*pi`. The raw
`quad_vec` absolute tolerance and error estimate are therefore converted by the
same `2*pi` factor.

## Microscopic evaluator contract

The active d-wave GK21 callback uses one common batched two-band q-workspace. The
batch implementation changes execution order only:

- shifted BdG Hamiltonians are assembled over the complete k batch;
- the two shifted eigensystems use stacked `numpy.linalg.eigh`;
- Peierls vector/contact vertices are generated from cached hopping arrays;
- all five `(rho,Jx,Jy,eta1,eta2)` band rotations are batched;
- direct/contact terms and all Ward-RHS pieces are reduced over k in NumPy.

There is no q-direction dispatch and no scalar runtime fallback. The scalar
q-workspace remains temporarily available only as a numerical equivalence
reference while real-`nk` performance is measured. Its retention or deletion is a
separate maintenance decision after the profile comparison.

The batch path must remain equivalent to the scalar reference for:

- shifted energies and occupations;
- left/right five-channel band vertices;
- direct contact contribution and phase direct term;
- `equal_forward`, `delta_v_mid`, `qM_mid`, and the final Ward RHS;
- complete response components at zero and positive Matsubara frequency.

No tolerance in those equivalence tests may be weakened to hide an algebraic or
storage-convention difference.

## Hard budget and failure semantics

The primary and audit passes share one hard cap on unique transverse evaluations,
defaulting to 256. Cache hits do not consume the cap. Complete microscopic orbit
points are counted separately.

A budget exception during a pass invalidates that pass. No partially accumulated
SciPy integral is exposed as a trusted estimate. If the audit exhausts the budget,
a completed primary estimate may be retained only as `diagnostic_only`.

Node-cap failure and wall-time failure are distinct:

- `unique_t > cap`: the integration representation/candidate failed the cost
  contract; optimizing the per-node evaluator does not fix the node count.
- `unique_t <= cap` but wall time is too high: profile and optimize the microscopic
  evaluator while preserving exact arithmetic and orbit coverage.

## Complete-orbit evaluator profiling

The primitive evaluator is a single reusable callable shared by the GK21 wrapper
and the profiling command. It records cumulative time in these stages:

- material midpoint workspace;
- q-dependent shifted workspace;
- Matsubara Kubo-factor construction;
- batched five-channel Kubo contraction;
- primitive-vector packing.

Orbit geometry/wrapping is timed separately by the complete-orbit workspace. A
single callback can be profiled with:

```bash
python -m validation matsubara dwave-orbit-evaluator-profile \
  --nk 1256 --mx 6 --my 4 --matsubara-indices 1 2 4 8
```

The profiler serializes the q-workspace implementation identifier. It is diagnostic
only: it does not perform a transverse integral or produce a Casimir-valid point.

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
- q-workspace implementation identifier;
- unique transverse evaluations, cache hits, and microscopic point evaluations;
- primary/audit error estimates, tolerances, ratios, status, and subinterval count;
- frozen group scales and observed-to-frozen ratios;
- primary/audit primitive-group differences;
- primary/audit sigma, reflection, and logdet differences;
- worst reported intervals;
- geometry, material-workspace, q-workspace, Kubo, packing, quadrature, and total
  wall times;
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

Failure at a q point must not create a q-specific integration rule. The outcome is
either acceptance of this common contract or a separately scoped redesign of the
common transverse/BZ representation.
