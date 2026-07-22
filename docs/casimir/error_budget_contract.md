# Full-Casimir error-budget and acceptance contract

This document records TODO item 4 for the long-lived production workflow branch.
The contract is pairing blind: `spm` and `dwave` may stop at different numerical
cutoffs, but they use the same tolerances, certificates, ladders and authorization
rules.

## Total error budget

For every pairing and physical case, the accepted free-energy error obeys the frozen
absolute/relative tolerance

```text
max(total_atol_J_m2, total_rtol * abs(finite_partial_J_m2)).
```

The policy serializes all nested fractions in the plan and therefore binds them into
the scientific-policy SHA:

- finite Matsubara terms: 0.7;
- omitted Matsubara tail: 0.3;
- finite outer-Q domain within each Matsubara term: 0.7;
- omitted outer-Q tail within each Matsubara term: 0.3;
- joint radial/angular estimate within the finite outer-Q budget: 0.8;
- offset audit within the finite outer-Q budget: 0.2;
- radial/angular split within the joint estimate: frozen by the plan, pairing blind.

The accumulated outer-Q error of all computed Matsubara terms and the omitted
Matsubara tail are added once. A result is not authorized unless the finite, tail and
total checks all pass.

## Outer-Q tail

The production route now attempts the passive-vacuum analytic bound at every finite
cutoff rather than waiting for the numerical shell ladder to be exhausted.

The analytic premise is recorded separately from the tail budget:

- positive Matsubara reflections use the passive-sheet/vacuum-admittance similarity
  theorem;
- the static reflection uses its exact spectral norm;
- the stored Frobenius norm is diagnostic only and is never an acceptance gate;
- only the active pairing/frequency scope is examined, so unrelated historical cache
  entries cannot reject the current run.

The numerical shell envelope is retained as an independently visible certificate and
diagnostic. Every cutoff record contains both analytic and geometric attempts and an
explicit rejection reason. The selected certificate path is written into each pairing
result.

## Matsubara tail

The legacy rule based on the last few individual term ratios is not a formal production
criterion. Production cutoffs must form complete dyadic blocks:

```text
0-1, 2-3, 4-7, 8-15, 16-31, 32-63, ...
```

For each block the controller forms the absolute envelope

```text
sum_n (abs(F_n) + certified_outer_error_n).
```

This prevents sign cancellation and remains meaningful when the physical term reaches
the outer-Q error floor. The final block in the configured tail window is an explicit
holdout. The block-ratio envelope, holdout ratio, finite-term error, omitted-tail
estimate and total budget must all pass. Per-term ratios are still emitted as diagnostic
telemetry but cannot authorize a result.

The numerical block envelope is a frozen acceptance contract, not a claim that a finite
window alone proves every possible asymptotic sequence. A later scientific change to
the estimator, ratio, block ladder or holdout policy changes the scientific-policy SHA.

## Status and authorization

The result distinguishes:

- numerical Matsubara convergence;
- formal policy passage;
- total error-budget closure;
- production authorization.

`production_casimir_allowed` is true only when all microscopic nodes, all outer-Q
certificates, the Matsubara-tail certificate and the total error budget pass. A
numerically converged result that is not formally authorized is written as
`diagnostic_only`, not `completed`.

## Fail-closed evidence

The contract tests cover:

- the known static Frobenius false negative;
- rejection of the legacy non-dyadic ladder;
- inverse-square and inverse-cube high-frequency tails;
- a holdout-block spike;
- explicit pairing-blind budget serialization;
- separation between numerical convergence and production authorization.

The historical frozen qualification commands remain available only for compatibility.
The formal production path is `python -m scripts.full_casimir plan/run` and uses the
certified controllers documented here.
