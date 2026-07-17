# Casimir adaptive angular outer-Q integration v1

## Status

This branch adds a production-owned, fail-closed global angular-order controller on
top of the adaptive radial controller. It does not replace or modify `run_casimir`
or the radial controller.

The public entry point is:

```python
from lno327.casimir import (
    AdaptiveAngularCasimirConfig,
    run_adaptive_angular_casimir,
)

result = run_adaptive_angular_casimir(AdaptiveAngularCasimirConfig())
```

A successful result remains a finite Matsubara partial result:

```text
status = adaptive_finite_partial
angular_converged = true
offset_audit_passed = true
outer_cutoff_fixed = true
outer_tail_estimated = false
matsubara_tail_estimated = false
production_casimir_allowed = false
```

## Frozen boundaries

This version keeps fixed:

- the finite radial interval `u in [0,u_max]`;
- the adaptive radial estimator and its tolerances;
- the full periodic angular domain `[0,2pi)`;
- the explicitly requested Matsubara indices;
- the microscopic model, transverse-N ladder, shifts and physical gates;
- the exact outer-Q measure and `u = 2 Q d` convention;
- the fixed controller and radial controller result contracts.

It does not infer `u_max`, estimate the omitted `u > u_max` tail, infer a
Matsubara cutoff, estimate the Matsubara tail, locally subdivide angular sectors,
differentiate torque, or authorize a production Casimir result.

## Angular-order ladder

The controller requires a strict doubling ladder:

```text
N_phi = 4 -> 8 -> 16 -> 32
```

For each order it runs the complete adaptive radial controller at one fixed primary
offset. A radial run must itself return:

```text
status = adaptive_finite_partial
radial_converged = true
all_microscopic_nodes_certified = true
```

before the angular comparison may be evaluated.

For every pairing and every requested Matsubara index, adjacent angular orders are
compared separately:

```text
absolute = |F_n(N_phi) - F_n(N_phi/2)|
scale = max(|F_n(N_phi)|, |F_n(N_phi/2)|)
tolerance = max(angular_atol, angular_rtol * scale)
passed = absolute <= tolerance
```

The total free energy is never used as a substitute for channelwise convergence.
Opposite-sign Matsubara contributions therefore cannot hide angular error by
cancellation.

The controller stops at the first order that supplies the configured number of
consecutive passing transitions. Exhausting the order ladder returns:

```text
status = unresolved
termination_reason = angular_order_ladder_exhausted
```

## Offset audit

After angular-order convergence, the controller reruns the full adaptive radial
calculation at the selected angular order using a distinct audit offset. The
default pair is:

```text
primary offset = 0.5
audit offset = 0.0
```

The offset comparison uses an independent absolute/relative tolerance and is also
required separately for every pairing and Matsubara contribution. A failed audit
returns:

```text
status = unresolved
angular_converged = true
offset_audit_passed = false
termination_reason = angular_offset_audit_failed
```

The two offsets must differ. No symmetry relation is assumed between them.

## Certified-point reuse

All angular-order and offset runs share one `CertifiedOuterQProvider`. Reuse occurs
only for bitwise-identical model-q coordinates under the same microscopic-policy
fingerprint.

For zero angular offset, doubling the periodic trapezoidal order nests the previous
angular nodes inside the new grid, so exact cache reuse is available wherever the
radial nodes also coincide. Half-cell grids are not assumed to be nested; any reuse
there remains exact-coordinate reuse only.

The provider cache still excludes outer panel boundaries, angular orders, angular
offsets and outer tolerances from its microscopic fingerprint. It never reuses a
point across changes to microscopic physics or certification policy.

## Fail-closed termination reasons

The angular controller can return unresolved for:

```text
radial_run_unresolved
angular_order_ladder_exhausted
offset_audit_radial_unresolved
angular_offset_audit_failed
point_provider_failure: ...
radial_result_contract_failure: ...
```

A missing, malformed, uncertified or non-finite inner result is never converted to
a zero angular contribution.

## Result and audit surface

The result records:

- the selected angular order;
- every primary-order radial run;
- adjacent-order absolute and relative differences;
- channelwise angular tolerances and pass flags;
- the independent offset-audit radial run;
- channelwise offset differences, tolerances and pass flags;
- final finite partial energies and radial error records;
- cumulative certified-point provider statistics;
- the explicit termination reason.

This controller remains diagnostic-only until angular, radial, outer-cutoff and
Matsubara-tail qualification are all established independently.
