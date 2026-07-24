# TODO 4: exact fast geometry assembly and qualification

## Status

Implementation branch: `feat/todo4-fast-geometry-assembly`

Pull request: `#37`

Current status: the core exact geometry plan, strict read-only batch executor,
distance reuse, scalar qualification, narrow real legacy replay runner, and
unit-aware reduced fixed-outer replay contract are implemented on the branch.
The representative qualification manifest and staged local runner are also
frozen. Representative real microscopic qualification records remain completion
items; this document does not authorize production Casimir output.

Qualification runbook:

```text
docs/casimir/todo4_representative_qualification_runbook.md
```

Frozen manifest and command:

```text
validation/configs/casimir/todo4_representative_v1.json
python -m validation diagnostic todo4-representative-qualification <action>
```

```text
cache_request_schema: material-response-cache-request-v1
geometry_batch_schema: material-geometry-batch-plan-v1
geometry_result_schema: material-geometry-batch-result-v1
qualification_schema: material-geometry-qualification-v1
legacy_replay_schema: material-geometry-legacy-replay-v1
outer_qualification_schema: material-geometry-outer-qualification-v1
valid_for_casimir_input: false
production_casimir_allowed: false
observable_error_budget_calibrated: false
```

## Purpose

TODO 2 separated a geometry-independent material response from reflection,
propagation, and logdet. TODO 3 made a response-level-certified
`MaterialResponseSnapshot` persistent and strictly addressable. TODO 4 consumes
those exact cached responses without microscopic fallback and makes angle and
distance assembly cheap enough to reuse.

The core chain is:

```text
GeometryBatchPlan
→ exact q_crystal response requirements
→ strict read-only cache preflight
→ unique plate reflections
→ unique distance-independent R1 @ R2 products
→ propagation/logdet updates for all distances
→ diagnostic equivalence reports
```

## Dependency direction

```text
material_response_cache_request.py
               |
               v
material_geometry_plan.py
               |
               v
material_geometry_batch.py
       /                     \
      v                       v
material_geometry.py   lifshitz_integrand.py
       \                     /
        v                   v
 prepared reflection / prepared passive pair
               |
               v
 distance-only LifshitzPoint evaluation

qualification-only modules:
material_geometry_qualification.py
        ├── scalar geometry replay
        └── matched legacy comparison
material_geometry_legacy_replay.py
        └── one explicit archived microscopic replay
material_geometry_outer_qualification.py
        └── already-matched arrays through one fixed outer reduction
material_geometry_qualification_campaign.py
        └── frozen representative plan construction
material_geometry_qualification_compatibility.py
        └── matched working-N / primary-shift preflight
material_geometry_qualification_execution.py
        └── staged populate / geometry / replay / verify orchestration
material_geometry_qualification_io.py
        └── atomic diagnostic artifacts
```

`material_response_cache_request.py` constructs exact TODO 3 request identities
without importing response integration orchestration. The geometry planner does
not read or write cache files. The batch executor requires a
`MaterialResponseCacheStore(mode="read_only")`; it does not import or call the
microscopic response engine. Legacy microscopic and fixed-outer operations are
quarantined to diagnostic qualification modules and cannot act as fallbacks.
The qualification campaign modules are imported only by the validation command;
core geometry never imports them.

## Exact geometry contract

The basis module remains the single source of truth:

```text
q_crystal = R(-theta_plate) @ q_lab
T_lab = R(theta_plate) @ T_crystal @ R(theta_plate).T
```

`GeometryBatchPlan` stores exact float64 laboratory momenta, plate angles, and
distances. Every plate requirement is built from the exact rotated `q_crystal`
and a complete TODO 3 response-cache identity.

Forbidden operations are explicit:

- no angle rounding;
- no q rounding;
- no nearest-q lookup;
- no symmetry q reduction;
- no interpolation or surrogate response;
- no rotation of a response evaluated at a different `q_crystal`.

The reflection adapter still checks that the loaded response momentum agrees
with `R(-theta_plate) @ q_lab`. Its tolerance is a consistency check, not a
cache-selection rule.

## Geometry plan

`build_geometry_batch_plan` accepts:

- one `MaterialResponseEngineConfig` defining exact material and certification
  request identity;
- labeled nonzero `q_lab` points;
- ordered unique `(theta_1, theta_2)` pairs;
- strictly increasing unique separations;
- geometry and passive-logdet tolerances.

The planner builds the q/frequency-independent material identity context once,
then constructs and deduplicates all exact plate response identities.

Distance and plate angle do not enter a material response identity. They do
enter geometry-plan identity because they define the requested assembly.
Runtime chunk size is excluded from both response and geometry scientific
identity because it does not change the numerical definition.

The frozen representative campaign intentionally uses a sparse matrix:

```text
5 geometry plans
16 geometry points
20 exact response identities
10 pairing/exact-q populate groups
32 distance updates
```

It covers SPM/d-wave, exact n=0/n=1, axial/oblique q, parallel/nonzero-relative
angle, and short/medium/long distance without constructing a full Cartesian
product.

## Strict cache preflight

`preflight_geometry_batch` requires a strict read-only TODO 3 store. It loads
each unique response identity at most once and returns all hits and misses.

A complete assembly requires zero misses. Missing entries raise
`GeometryBatchCacheIncomplete`; the executor does not:

- switch to populate mode;
- write cache artifacts;
- call `evaluate_material_response_ladder`;
- call arbitrary-q microscopic integration;
- fall back to the archived point route.

The preflight retains complete certified artifacts, including working/audit N,
primary shift, certification evidence, and exact identity contracts. These are
used only by later qualification modules to prove matched numerical evidence.

The representative runner adds a second hard preflight before legacy replay. It
requires both plates of every planned point to use the same working N and exact
primary shift. Incompatibility is written to `legacy_compatibility.json`; the
runner does not silently search for a common N or shift.

## Prepared reflection and distance reuse

For each exact `(response identity, q_lab, theta_plate)` requirement the batch
executor constructs one plate reflection. Repeated use by multiple angle pairs
or distances reuses that reflection.

`prepare_passive_sheet_pair` then checks reflection compatibility and computes
once:

```text
R_product = R1 @ R2
lambda_product = eigenvalues(R_product)
kappa
passivity / real-spectrum audit
```

For every separation only the following quantities are updated:

```text
p(d) = exp(-2 kappa d)
M(d) = I - p(d) R_product
logdet(d) = sum_alpha log1p(-p(d) lambda_alpha)
```

The scalar `passive_sheet_logdet` delegates to the same prepare/evaluate
implementation, preventing scalar and batch algebra from drifting apart.

## Structural performance counters

`GeometryBatchResult.metadata` records deterministic operation counts rather
than using wall time as a correctness gate:

```text
response_load_count = unique exact response identities
reflection_build_count = unique response/q_lab/theta requirements
prepared_pair_count = q/frequency/angle-pair points
distance_update_count = prepared_pair_count * number of distances
microscopic_integration_call_count = 0
response_certification_call_count = 0
cache_write_count = 0
```

Wall-time measurements may be runtime telemetry, but cannot change plan or
response identity and are not formal acceptance gates.

## Qualification contracts

### Scalar versus batch

`qualify_batch_point_against_scalar` recomputes every requested distance through
the existing scalar `assemble_two_plate_logdet` route using the same persisted
responses. It compares signed logdet and trace-log matrices under one explicit
dimensionless absolute/relative policy.

### Archived point route versus persisted batch

`qualify_matched_legacy_point` is a quarantined comparison boundary.
`run_matched_legacy_geometry_replay` is the explicit narrow runner that rebuilds
one archived point. It obtains working N and primary shift from the two certified
artifacts and performs exactly one old-route integration; it does not search an
N ladder or populate caches.

A legacy point is admitted only when all of the following match the persisted
primary responses:

- pairing, model, finite temperature, Matsubara index and exact frequency;
- exact `q_crystal` for both plates and exact plate angles;
- material-state and response-policy identities;
- primitive contract and phase-Hessian policy;
- working N and exact primary BZ shift;
- canonical reduction block size;
- every physical tolerance that the archived helper can express.

The comparison checks each plate reflection matrix, the two-plate product,
product eigenvalues, signed logdet, exact q mapping, and legacy hard physical
closure. Any identity, N/shift/reduction, primitive, phase, or policy mismatch
fails before accepting numerical equivalence.

### Reduced fixed-outer replay

`qualify_fixed_outer_geometry_replay` consumes two already-matched arrays with
shape `(Matsubara index, fixed outer node)` and sends both through the same
`OuterQPolarGrid` and finite Matsubara reduction. It does not evaluate material
responses or geometry.

`FixedOuterEquivalencePolicy` uses separate absolute tolerances for:

```text
dimensionless node logdet
outer integral in m^-2
Matsubara contribution in J/m^2
total finite partial sum in J/m^2
```

One dimensionful absolute tolerance is never reused for quantities with
different units. This replay remains a finite, tail-free diagnostic partial sum
and is not an observable-level error budget.

## Representative qualification execution

The validation-only runner exposes strictly separated stages:

```text
plan
→ preflight
→ populate
→ preflight --require-complete
→ geometry
→ legacy
→ verify
```

Only `populate` can call the response ladder or write certified response cache
entries. `geometry`, the new-route side of `legacy`, and `verify` reopen the
store in strict read-only mode. All reports are atomic and record source commit,
manifest SHA and frozen plan SHA.

The local command sequence, shard policy, output layout and stop conditions are
specified in `todo4_representative_qualification_runbook.md`.

## Implemented tests

The branch covers:

- zero and positive Matsubara scalar/batch equivalence;
- multi-distance prepared-pair equivalence;
- exact angle mapping and exact requirement deduplication;
- distance changes leaving material requirements unchanged;
- runtime chunk changes leaving scientific identities unchanged;
- one material identity-context construction per plan;
- strict read-only misses returning all missing identities;
- no cache creation or microscopic fallback on misses;
- response, reflection, prepared-pair, and distance-update counters;
- scalar qualification reports;
- individual reflection, product, eigenvalue, and logdet comparison contracts;
- matched legacy material, primitive, phase, N, shift, and reduction gates;
- one-N/one-shift legacy replay orchestration with no ladder search or cache write;
- unit-aware fixed-outer replay pass/fail tests;
- frozen representative manifest and deterministic exact-response counts;
- empty-cache fail-closed behavior and deterministic shard partitioning;
- legacy working-N/primary-shift readiness preflight;
- atomic frozen-plan conflict rejection;
- repository-level dependency and qualification-campaign quarantine guards.

## Completion items

TODO 4 is not marked complete until the branch also contains reviewed,
reproducible diagnostic records for representative real microscopic points:

```text
pairing: spm and dwave
frequency: exact n=0 and at least one positive Matsubara index
q: axial and oblique representative points
angle: zero and nonzero relative orientation
distance: representative short / medium / long values
```

For each point the archived and new routes must use matched complete contracts
and pass reflection/product/logdet comparison. A small fixed-outer-Q set must
then provide real old/new logdet arrays to the unit-aware reduced replay.

These are narrow qualification runs, not a broad production scan. Failure or
unresolved response certification must remain explicit; the qualification set
must not silently expand to search for favorable points.

## Explicit exclusions

TODO 4 does not implement:

- frequency compression, reference subtraction, Chebyshev, IR, or DLR;
- q or angle interpolation and surrogate response libraries;
- Matsubara-tail certification changes;
- broad angle-distance production scans;
- observable-level free-energy, pressure, or torque error budgets;
- migration or promotion of archived point caches;
- true zero-temperature microscopic calculation;
- production admission.
