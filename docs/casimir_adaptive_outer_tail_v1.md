# Casimir adaptive outer-Q cutoff and tail v1

## Status

This branch adds a production-owned, fail-closed outer-cutoff controller on top of
the joint radial-angular finite-domain controller. It extends a cumulative
`u_max` ladder and establishes a channelwise envelope for the omitted
`u > u_max` contribution.

The public entry point is:

```python
from lno327.casimir import (
    AdaptiveOuterTailCasimirConfig,
    run_adaptive_outer_tail_casimir,
)

result = run_adaptive_outer_tail_casimir(AdaptiveOuterTailCasimirConfig())
```

A successful result is still only a finite Matsubara partial result:

```text
status = adaptive_finite_partial
cutoff_converged = true
outer_cutoff_adaptive = true
outer_tail_estimated = true
matsubara_tail_estimated = false
production_casimir_allowed = false
```

## Frozen boundaries

This version does not modify:

- `run_casimir`;
- the radial parent-versus-children estimator;
- the global angular-order ladder or offset audit;
- the microscopic model, transverse-N ladder, shifts, or physical gates;
- the explicitly requested finite Matsubara index set;
- the exact measure and `u = 2 Q d` convention.

It does not infer or estimate the Matsubara tail, locally subdivide angular
sectors, differentiate torque, or authorize a production Casimir result.

## Cumulative cutoff ladder

The default cumulative panel boundaries are:

```text
u = 0, 6, 10, 14, 18, 24, 30, 36, 42
```

For cutoff index `k`, the wrapped joint controller receives:

```text
initial_panel_edges = (0, u_1, ..., u_k)
```

Extending the cutoff therefore preserves every previous top-level panel and adds
only the new shell `[u_{k-1}, u_k]`. All cutoff runs share one
`CertifiedOuterQProvider`; reuse remains restricted to bitwise-identical model-q
coordinates under the same microscopic-policy fingerprint.

Every cumulative finite-domain run must itself return:

```text
status = adaptive_finite_partial
joint_converged = true
radial_budget_passed = true
angular_budget_passed = true
offset_audit_passed = true
all_microscopic_nodes_certified = true
```

Any unresolved inner run stops cutoff extension immediately. No shell
contribution or tail estimate is inferred from an unresolved cumulative result.

## Total outer-Q error budget

For every pairing and Matsubara contribution, the controller defines:

```text
T_total = max(total_outer_atol, total_outer_rtol * |F_n|)
T_finite = finite_domain_fraction * T_total
T_tail = tail_fraction * T_total
finite_domain_fraction + tail_fraction = 1
```

The finite-domain allocation is split again between the joint radial-angular
comparison and the independent offset audit:

```text
joint_fraction_within_finite + offset_fraction_within_finite = 1
```

The wrapped joint controller receives the correspondingly scaled absolute and
relative tolerances. The finite-domain error evidence recorded by this layer is:

```text
E_finite = E_joint + E_offset
```

where `E_joint` is the existing radial-plus-angular channelwise estimate and
`E_offset` is the selected-order offset difference. The independent audit radial
errors remain hard gates inside the joint controller.

## Shell records

Let `F_k` be the accepted cumulative finite-domain contribution at cutoff `u_k`,
and let `E_k` be its finite-domain error evidence. The new shell record is:

```text
Delta F_k = F_k - F_{k-1}
E_shell,k = E_k + E_{k-1}
A_k = |Delta F_k| + E_shell,k
```

The absolute envelope amplitude `A_k` is used rather than the signed shell
contribution. Sign oscillation therefore cannot create a false small tail through
cancellation. All quantities are stored separately for every pairing and every
Matsubara index.

## Tail window contract

Tail inference begins only for shells whose left boundary satisfies:

```text
left_u >= tail_start_u
```

The final `tail_window_shells` shells must have equal widths within the configured
absolute and relative width tolerances. This is required because the geometric
envelope extrapolates future shells of the same width.

For consecutive envelope amplitudes:

```text
r_j = A_j / A_{j-1}
```

with the exact `0/0 -> 0` convention. Every observed ratio in the window must
satisfy:

```text
r_j <= tail_ratio_max < 1
```

The conservative geometric tail bound is then:

```text
E_tail <= A_last * tail_ratio_max / (1 - tail_ratio_max)
```

The configured ratio cap, rather than the smaller observed ratio, is used in the
bound.

## Channelwise acceptance

For every pairing and Matsubara index, all of the following must hold:

```text
observed decay ratios <= tail_ratio_max
E_finite <= T_finite
E_tail <= T_tail
E_finite + E_tail <= T_total
```

The total free energy is never used as a substitute for channelwise evidence.
Opposite-sign Matsubara contributions or pairing cancellations cannot hide a
non-decaying shell sequence.

The controller stops at the first cutoff satisfying every channel. Exhausting the
cutoff ladder without establishing all gates returns `unresolved`.

## Fail-closed termination reasons

The controller can return:

```text
finite_domain_run_unresolved: ...
outer_tail_microscopic_q_node_budget_exhausted
outer_tail_window_not_established
outer_tail_shell_width_contract_failed
outer_tail_decay_ratio_not_established
outer_tail_budget_not_met
total_outer_budget_not_met
outer_cutoff_ladder_exhausted
point_provider_failure: ...
outer_tail_result_contract_failure: ...
```

Missing, malformed, uncertified, or non-finite inner results are never converted
to zero shell contributions.

## Result and audit surface

The result records:

- every attempted cumulative cutoff;
- the exact cumulative panel edges supplied to each joint run;
- every shell contribution and its quadrature-error bound;
- shell envelope amplitudes and observed ratios;
- the configured ratio envelope and equal-width audit;
- finite-domain, tail, and total error estimates and tolerances;
- pairing- and Matsubara-resolved pass flags;
- the selected `u_max`;
- cumulative provider/cache statistics;
- the explicit termination reason.

This controller remains non-production because the Matsubara cutoff and high
frequency tail are still unresolved.
