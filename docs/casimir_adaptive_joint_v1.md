# Casimir joint radial-angular outer-Q control v1

## Status

This branch adds a production-owned, fail-closed controller that coordinates the
existing adaptive radial estimator and global periodic angular-order ladder. It does
not replace or modify `run_casimir`, the radial controller, or the angular controller.

The public entry point is:

```python
from lno327.casimir import (
    AdaptiveJointCasimirConfig,
    run_adaptive_joint_casimir,
)

result = run_adaptive_joint_casimir(AdaptiveJointCasimirConfig())
```

A successful result remains a finite Matsubara partial result:

```text
status = adaptive_finite_partial
joint_converged = true
radial_budget_passed = true
angular_budget_passed = true
offset_audit_passed = true
outer_cutoff_fixed = true
outer_tail_estimated = false
matsubara_tail_estimated = false
production_casimir_allowed = false
```

## Frozen boundaries

The controller keeps fixed:

- the finite radial interval `u in [0,u_max]`;
- the full periodic angular domain `[0,2pi)`;
- the explicitly requested Matsubara indices;
- the microscopic model, transverse-N ladder, shifts, and physical gates;
- the exact outer-Q measure and `u = 2 Q d` convention;
- the existing radial and angular estimator definitions.

It does not infer `u_max`, estimate the omitted `u > u_max` tail, infer a Matsubara
cutoff, estimate the Matsubara tail, locally subdivide angular sectors, differentiate
torque, or authorize a production Casimir result.

## One joint finite-domain budget

For every pairing and every Matsubara contribution, define the finite-domain outer-Q
tolerance

```text
T_outer = max(outer_atol, outer_rtol * scale)
scale = max(|F(N_phi/2)|, |F(N_phi)|)
```

The user supplies positive radial and angular fractions that sum to one:

```text
f_r + f_phi = 1
```

The allocated component tolerances are

```text
T_r   = f_r   * T_outer
T_phi = f_phi * T_outer
```

The radial allocation is split equally between the two radial calculations entering
an adjacent angular-order comparison. Each radial run therefore receives

```text
per-run radial fraction = f_r / 2
```

through its radial relative and absolute tolerances.

## Conservative comparison error

For adjacent angular orders, the controller uses

```text
E_r = E_r(N_phi/2) + E_r(N_phi)
E_phi = |F(N_phi) - F(N_phi/2)|
E_joint = E_r + E_phi
```

and requires separately

```text
E_r <= T_r
E_phi <= T_phi
E_joint <= T_outer
```

for every pairing and Matsubara index. The total free energy is never used as a
substitute for these channelwise gates. Opposite-sign channels cannot hide either
radial or angular error by cancellation.

The two component scores are

```text
S_r   = max_channels(E_r / T_r)
S_phi = max_channels(E_phi / T_phi)
```

The controller advances the direction with the larger normalized score. Exact ties
use the configured deterministic tie break, radial by default.

## Direction changes

The first state compares the first two members of the strict angular doubling ladder.
At a fixed angular pair and radial round cap:

```text
S_r > S_phi   -> increase the radial refinement-round cap
S_phi > S_r   -> advance to the next angular order
```

Increasing the radial cap reruns the same adjacent angular-order pair with tighter
radial evidence. Advancing the angular direction shifts the comparison window from

```text
N_phi/2 -> N_phi
```

to

```text
N_phi -> 2 N_phi
```

at the current radial cap.

Certified microscopic points are shared through one `CertifiedOuterQProvider`. The
controller also caches identical in-process radial runs keyed by angular order,
offset, and radial round cap.

## Certified unresolved radial estimates

A radial run that stops only because its current refinement-round cap was reached may
still provide a certified finite estimate and radial error bound. The joint controller
may use that evidence to decide which direction should advance next.

It never consumes a radial result with unresolved microscopic points, malformed or
non-finite channel arrays, exhausted microscopic-node budget, or another hard inner
failure. Such states return `unresolved` immediately.

Panel-depth exhaustion is also fail-closed when radial refinement remains the selected
direction.

## Offset audit

After the radial and angular allocations both pass, the controller evaluates the
selected angular order at the independent audit offset.

The primary and audit radial errors are checked against the radial allocation, while
the offset difference uses its own absolute/relative tolerance. If the audit is
radial-error dominated, the controller raises the radial round cap. If offset
sensitivity dominates, it advances angular order. Failure at the maximum angular
order remains unresolved.

## Resource budgets

The controller enforces:

- the radial controller's maximum refinement-round ceiling;
- maximum joint decision iterations;
- maximum total unique microscopic q nodes across all orders and offsets;
- each radial run's own panel-depth and q-node limits;
- strict angular-order doubling.

Budget exhaustion never converts an unevaluated region or point into zero.

## Result and audit surface

The result records:

- the selected angular order and radial round cap;
- every radial run and its exact order, offset, and round cap;
- every direction decision and both normalized component scores;
- per-pairing, per-Matsubara radial, angular, joint, and offset errors;
- allocated component tolerances and pass flags;
- final finite partial energies;
- cumulative provider cache and certification statistics;
- the explicit termination reason.

Representative unresolved reasons include:

```text
radial_run_unresolved
radial_panel_depth_exhausted
joint_radial_round_budget_exhausted
joint_angular_order_ladder_exhausted
joint_offset_audit_failed_at_maximum_angular_order
joint_iteration_budget_exhausted
joint_microscopic_q_node_budget_exhausted
point_provider_failure
radial_result_contract_failure
```

This controller remains diagnostic-only until finite-domain joint control, automatic
outer cutoff and tail control, Matsubara cutoff and tail control, and observable-level
qualification are all established independently.
