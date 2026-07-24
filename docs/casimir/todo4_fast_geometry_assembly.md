# TODO 4: exact fast geometry assembly and qualification

## Status

Implementation branch: `feat/todo4-fast-geometry-assembly`

Pull request: `#37`

Current status: core exact geometry planning and strict read-only batch assembly are
implemented on the branch. Scalar/batch equivalence and a matched legacy
qualification boundary are implemented. Representative real microscopic
old/new qualification records and reduced outer-Q replay remain completion
items; this document does not authorize production Casimir output.

```text
geometry_batch_schema: material-geometry-batch-plan-v1
geometry_result_schema: material-geometry-batch-result-v1
qualification_schema: material-geometry-qualification-v1
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

The intended chain is:

```text
GeometryBatchPlan
→ exact q_crystal requirements
→ strict read-only cache preflight
→ unique plate reflections
→ unique distance-independent R1 @ R2 products
→ propagation/logdet updates for all distances
→ diagnostic equivalence reports
```

## Dependency direction

```text
material_response_cache_identity / cached response request identity
                         |
                         v
              material_geometry_plan.py
                         |
                         v
              material_geometry_batch.py
                  /               \
                 v                 v
       material_geometry.py   lifshitz_integrand.py
                 \                 /
                  v               v
          prepared reflection / prepared sheet pair
                         |
                         v
          distance-only LifshitzPoint evaluation

material_geometry_qualification.py
        ├── scalar geometry replay
        └── quarantined legacy point comparison
```

The core planner does not read or write cache files. The core batch executor
requires a `MaterialResponseCacheStore(mode="read_only")`; it does not import or
call the microscopic response engine. The legacy point engine is imported only
inside the qualification module.

## Exact geometry contract

The basis module remains the single source of truth:

```text
q_crystal = R(-theta_plate) @ q_lab
T_lab = R(theta_plate) @ T_crystal @ R(theta_plate).T
```

`GeometryBatchPlan` stores exact float64 laboratory momenta, plate angles, and
distances. Every plate response requirement is built from the exact rotated
`q_crystal` and a complete TODO 3 response-cache identity.

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

- one `MaterialResponseEngineConfig` defining the exact material and
  certification request identity;
- labeled nonzero `q_lab` points;
- ordered unique `(theta_1, theta_2)` pairs;
- strictly increasing unique separations;
- geometry and passive-logdet tolerances.

The planner builds the q/frequency-independent material identity context once,
then constructs and deduplicates all exact plate response identities.

Distance and plate angle do not enter a material response identity. They do
enter the geometry plan identity because they define the requested assembly.
Runtime chunk size is excluded from both response and geometry scientific
identity because it does not change the numerical definition.

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
primary shift, and certification evidence, so later legacy qualification can
verify that comparisons use matched numerical evidence.

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

Wall-time measurements may be added as runtime telemetry, but cannot change plan
or response identity and are not formal acceptance gates.

## Qualification contracts

### Scalar versus batch

`qualify_batch_point_against_scalar` recomputes every requested distance through
the existing scalar `assemble_two_plate_logdet` route using the same persisted
responses. It compares signed logdet and trace-log matrices under one explicit
absolute/relative policy.

### Archived point route versus persisted batch

`qualify_matched_legacy_point` is a quarantined diagnostic boundary. A legacy
point is admitted for comparison only when all of the following match the
persisted primary responses:

- pairing, temperature, frequency, q and plate angles;
- exact `q_crystal` for both plates;
- working N;
- exact primary BZ shift;
- canonical reduction block size;
- physical response policy and trace-log geometry policy supplied by the caller.

The comparison checks the two-plate product matrix, product eigenvalues, signed
logdet, exact q mapping, and legacy hard physical closure. A mismatch in the
N/shift/reduction contract fails before numerical comparison.

The archived route remains diagnostic. New geometry execution never falls back
to it.

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
- matched legacy N/shift/reduction qualification gates;
- repository-level dependency guards.

## Completion items

TODO 4 is not marked complete until the branch also contains reviewed,
reproducible diagnostic qualification records for representative real
microscopic points:

```text
pairing: spm and dwave
frequency: exact n=0 and at least one positive Matsubara index
q: axial and oblique representative points
angle: zero and nonzero relative orientation
distance: representative short / medium / long values
```

For each point the archived and new routes must use matched N/shift/reduction and
pass the direct reflection/product/logdet comparison. A small fixed-outer-Q
replay must then confirm that organizing the same qualified point values in a
batch does not alter the reduced outer integral.

These are narrow qualification runs, not a broad production scan.

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
