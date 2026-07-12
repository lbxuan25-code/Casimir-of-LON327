# d-wave adaptive integration after finite-q longitudinal diagnosis

## Status

The exact-static d-wave validation path has returned to global vector-valued
iterated-adaptive integration.  Commensurate uniform grids remain the structural
Ward reference, not the production-convergence method.

All results remain diagnostic-only:

```text
diagnostic_only = True
projection_applied = False
production_reference_established = False
valid_for_casimir_input = False
```

## Correct assembly order

For one fixed physical q, the adaptive path now performs:

1. evaluate the complete 48-component primitive density at common adaptive nodes;
2. integrate electromagnetic, collective, q=0 Goldstone-counterterm, phase-direct,
   and analytic Ward-RHS channels with the same vector quadrature;
3. assemble the complete Brillouin-zone primitive response;
4. apply the diagnosed nearest-neighbour phase-Hessian pullback

   ```text
   K_HS[1,1](q) = g_bond(q) K_HS[1,1](0)
   g_bond(q) = [cos(qx/2)^2 + cos(qy/2)^2] / 2
   ```

   while leaving K_HS[0,0], K_HS[0,1], and K_HS[1,0] unchanged;
5. rebuild exactly one amplitude/phase Schur complement;
6. validate the adaptive analytic Ward RHS with both the generic RHS-aware audit
   and `strict_static_q_normalized_v1`;
7. extract raw chi_bar and Dbar_T without longitudinal projection.

The q-dependent metric is deliberately not inserted into the pointwise adaptive
integrand.  It is a geometric pullback of the already-integrated scalar Goldstone
counterterm and must be applied exactly once after primitive assembly.

## Public commands

Single adaptive level with both nesting orders:

```bash
python -m validation static dwave-iterated-adaptive ...
```

Resumable convergence over successively tighter adaptive levels:

```bash
python -m validation static dwave-adaptive-convergence \
  --levels \
    coarse:1e-5:2e-3:60000:120:120 \
    medium:3e-6:1e-3:120000:160:160 \
    fine:1e-6:5e-4:240000:200:200 \
  ...
```

Each level runs both `xy` and `yx` unless explicitly overridden.  The convergence
gate requires:

- the finest level to complete both adaptive orders;
- the finest level to pass strict static Ward closure in both orders;
- the finest level to pass raw static sheet validation in both orders;
- xy/yx chi_bar and Dbar_T agreement within the order tolerance;
- the finest two adaptive levels to agree in chi_bar and Dbar_T within the
  observable tolerance for every order.

## Role of the other integration paths

### Commensurate uniform grids

Retained as the exact discrete Ward-structure reference.  They established the
bond metric, complementary half-step treatment, direction coverage, and C4
covariance.  They do not define production quadrature convergence.

### Periodic shift ensemble

Retained only as an independent cross-check or possible future throughput
alternative.  It must first be compared against an adaptive reference and does
not establish that reference itself.

## Verification

Regression tests cover:

- policy-aware adaptive postprocessing versus the optimized q-workspace on the
  same complete periodic quadrature;
- unchanged amplitude and cross counterterm entries;
- changed phase diagonal only;
- rejection of double bond-metric application;
- public CLI routing through the corrected adaptive command;
- adaptive convergence-level parsing and per-order cross-level metrics.
