# TODO 2: finite-temperature material response / geometry boundary

## Status

Implementation branch: `refactor/material-response-boundary`

This document records the new geometry-independent response path and the
remaining migration boundary. It does **not** authorize a production Casimir
calculation.

```text
valid_for_casimir_input: false
production_casimir_allowed: false
observable_error_budget_calibrated: false
persistent_response_cache_present: false
```

The archived zero-degree campaign is not resumed or inherited by this work.

## Ownership and dependency direction

The intended one-way dependency graph is:

```text
arbitrary-q microscopic integration
        |
        v
material_response.py
        |
        +--> material_response_certification.py
        |          |
        |          v
        |    CertifiedMaterialResponse (diagnostic only)
        |
        v
material_geometry.py
        |
        v
material_two_plate.py
        |
        v
observable / outer integration layers (future migration)
```

The clean response-ladder orchestration is implemented in
`material_response_engine.py`. It accepts crystal-frame momentum and material
numerical policy only. It does not accept laboratory momentum, plate angle,
separation, reflection policy, or outer quadrature state.

## Module contracts

### `material_response.py`

Owns the single authoritative conversion:

```text
ArbitraryQPeriodicBZResult
  -> EffectiveEMKernel
  -> RHS-aware Ward validation
  -> zero- or positive-Matsubara sheet response
  -> MaterialResponseSample
```

The exact-zero path extracts static susceptibility/stiffness and never divides
by frequency. The positive-frequency path constructs the crystal-xy sheet
conductivity. The two sectors are type-checked at object construction.

### `material_response_certification.py`

Owns N/shift convergence in response space:

- exact compatibility of `q_crystal`, frequency, and sector;
- separate `chi_bar` and `dbar_t` checks at zero Matsubara frequency;
- spectral-norm comparison of `sigma_tilde_xy` at positive frequency;
- complete pairwise cross-shift checks;
- adjacent-N checks for each shift;
- complete pairwise multi-N/multi-shift oscillatory envelope;
- deterministic audit-shift selection.

A successful result has status `response_certified_diagnostic`. It is not a
production-admission token.

### `material_response_engine.py`

Owns the geometry-free N ladder for one pairing and one exact crystal momentum.
It batches all active Matsubara frequencies for each N/shift microscopic call,
constructs response samples, and stops each frequency independently after its
response-space convergence contract is established.

### `material_geometry.py`

Owns one-plate reflection assembly from a precomputed material response. It has
no microscopic fallback and checks the exact relation between the supplied
crystal response and requested laboratory geometry through the existing
reflection adapters.

### `material_two_plate.py`

Owns two-plate reflection, propagation, and signed passive logdet assembly from
two supplied material responses. It never evaluates or certifies microscopic
response.

## Stable engineering rules

1. There is one material conversion implementation. Compatibility code must call
   it rather than copy its formulas.
2. Material policy and geometry policy are distinct frozen data contracts.
3. Material modules must not import reflection, Lifshitz, or outer integration.
4. Geometry modules must not import microscopic integration or response
   certification engines.
5. Crystal momentum is compared exactly; no rounding, wrapping, nearest-node
   substitution, or interpolation is permitted in TODO 2.
6. Runtime scheduling and telemetry are not material identity.
7. Zero and positive Matsubara sectors remain distinct paths.
8. Every failure is fail-closed. No silent legacy fallback is permitted.
9. All new response and certification objects remain diagnostic-only.

## Numerical equivalence contract

The geometry assembly tests compare the new path directly against the existing
reflection and passive-logdet functions for both zero and positive Matsubara
sectors. The new path must reproduce the same matrices and logdet when given the
same material response and geometry policy.

This proves formula preservation at the response/geometry boundary. It does not
qualify response-space tolerances against a final observable error budget.

## Legacy route and migration boundary

`fixed_transverse_point_engine.py` and
`fixed_transverse_point_certification.py` still implement the archived
geometry-specific `two_plate_logdet` convergence route. The new TODO 2 modules do
not silently replace that route.

Until a separate migration commit is qualified:

- the legacy fixed route remains available only for historical regression and
  archived diagnostic interpretation;
- the new material-response engine is the authoritative implementation target
  for TODO 2 and later reusable response work;
- no old point cache is reclassified as a reusable material-response cache;
- no old logdet sweet spot is reclassified as response certification.

## Explicit exclusions

The following belong to later TODO items and are intentionally absent here:

- persistent atomic response-cache format and cache-key contract (TODO 3);
- broad angle/distance batch orchestration over a persisted response library
  (TODO 4);
- frequency interpolation or compression (TODO 5);
- observable-level error allocation and production admission (TODO 10);
- restoration or continuation of the stopped old campaign.

## TODO 2 completion gate

TODO 2 may be marked complete only after all of the following hold:

- geometry-free material response types and builder are stable;
- response-space N/shift certification is pairwise complete and fail-closed;
- clean geometry-free response-ladder orchestration is tested;
- one- and two-plate geometry assembly have no microscopic fallback;
- zero- and positive-frequency equivalence tests pass;
- architecture dependency tests pass;
- repository-wide contract tests pass;
- documentation identifies the legacy route and does not present it as the new
  reusable-response architecture;
- production authorization remains false.
