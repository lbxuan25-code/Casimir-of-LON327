# d-wave finite-q phase-Hessian core implementation

## Status

The analytically derived nearest-neighbour bond phase metric is now available
as an explicit core response policy.  It remains opt-in and fail-closed.

```text
phase_hessian_policy = "q_independent"                  # default
phase_hessian_policy = "nearest_neighbor_bond_metric"  # diagnostic opt-in
```

The default policy is an exact numerical no-op.  Existing response arrays and
Schur kernels are not recomputed or changed; only explicit policy metadata is
added at the public response boundary.

## Core formula

For the d-wave bond-endpoint gauge tangent,

\[
g_{\rm bond}(q)
=\frac{1}{2}\left[
\cos^2\!\left(\frac{q_x}{2}\right)
+\cos^2\!\left(\frac{q_y}{2}\right)
\right].
\]

The opt-in policy changes only

\[
K_{22}^{\rm HS}(q)
=g_{\rm bond}(q)K_{22}^{\rm HS}(0).
\]

It preserves `K_11`, `K_12`, and `K_21` exactly and rebuilds

\[
K_{\eta\eta}=K_{\eta\eta}^{\rm bubble}+K_{\eta\eta}^{\rm HS},
\qquad
K_{\rm eff}=K_{SS}-K_{S\eta}K_{\eta\eta}^{-1}K_{\eta S}.
\]

## Core module

The shared implementation lives in

```text
src/lno327/response/phase_hessian.py
```

It provides:

- `PhaseHessianPolicy`;
- `nearest_neighbor_dwave_bond_metric`;
- `apply_phase_hessian_policy_to_components`;
- `finite_q_bdg_response_from_model_ansatz_with_phase_hessian`;
- explicit metadata and an application audit record.

The policy rejects unsupported combinations.  The bond metric currently
requires:

```text
ansatz.name == "dwave"
ansatz.phase_vertex == "bond_endpoint_gauge"
collective_mode == "amplitude_phase"
```

## Public routing

`FiniteQEngineOptions` now contains

```python
phase_hessian_policy: PhaseHessianPolicy = "q_independent"
```

The policy is applied at both public response boundaries:

1. the direct model-driven finite-q workflow;
2. the optimized `FiniteQQWorkspace` evaluation functions exposed through
   `lno327.response.workspace` and `lno327.response`.

The material workspace continues to cache the q=0 Goldstone counterterm once.
The q-dependent multiplier is applied only when one q workspace is evaluated,
so the material cache remains q-independent.

The validation full-kernel audit now delegates to the same core helper rather
than maintaining a second implementation.

## Metadata

An opt-in corrected response records at least:

```text
phase_hessian_policy
phase_hessian_policy_opt_in
phase_hessian_multiplier
phase_hessian_base_counterterm_22
phase_hessian_applied_counterterm_22
phase_hessian_counterterm_delta_22
phase_hessian_changed_only_22
phase_hessian_source
```

It also updates the collective condition number and inverse method after the
new Schur complement is formed.

All opt-in responses retain:

```text
diagnostic_only = True
projection_applied = False
production_reference_established = False
valid_for_casimir_input = False
```

## Tests

The implementation tests verify:

1. policy-name validation and the exact bond metric;
2. the default q-independent policy preserves all numerical response arrays;
3. only the phase counterterm diagonal changes under the bond policy;
4. unsupported pairing ansatzes and phase vertices fail closed;
5. direct and optimized q-workspace evaluations use the same policy and agree
   numerically;
6. legacy validation metadata aliases remain available;
7. existing dependency-boundary contracts remain intact.

## Remaining promotion gate

Core availability does not establish a production reference.  Promotion still
requires a q/direction/grid validation family with explicit absolute
longitudinal residual gates, including both half-q-compatible grids and
componentwise complementary-subgrid averages for odd commensurate shifts.
