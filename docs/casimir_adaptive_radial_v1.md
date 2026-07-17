# Casimir adaptive radial outer-Q integration v1

## Status

This branch adds a production-owned, fail-closed adaptive radial integrator beside
the qualified fixed controller. It does not replace or modify `run_casimir`.

The public entry point is:

```python
from lno327.casimir import (
    AdaptiveRadialCasimirConfig,
    run_adaptive_radial_casimir,
)

result = run_adaptive_radial_casimir(AdaptiveRadialCasimirConfig())
```

A successful result remains a finite Matsubara partial result:

```text
status = adaptive_finite_partial
radial_converged = true
outer_cutoff_fixed = true
outer_tail_estimated = false
matsubara_tail_estimated = false
production_casimir_allowed = false
```

## Frozen boundaries

The first adaptive version changes only the radial panel partition on the fixed
finite interval `u in [0, u_max]`. It keeps fixed:

- the full periodic angular rule and angular offset;
- the explicitly requested Matsubara indices;
- the microscopic model and point-specific transverse-N certification;
- the shift set, Ward gates, static gates, reflection gates and logdet gates;
- the physical outer-Q measure and `u = 2 Q d` convention;
- the fixed controller and its golden reference.

It does not infer an outer cutoff, estimate the omitted `u > u_max` tail, estimate
the Matsubara tail, adapt the angular grid, differentiate torque, or authorize a
production Casimir result.

## Incremental certified-point provider

`CertifiedOuterQProvider` delegates every new point to the existing production
transverse-point certifier. It owns no microscopic numerical implementation.

Each point is keyed by:

```text
(pairing, Matsubara n, float(qx).hex(), float(qy).hex())
```

Only bitwise-identical model-q points are reused. The optional JSON cache is atomic,
contains the complete point-certification payload, and is protected by a fingerprint
of every input that can change a microscopic point. Outer panel boundaries and
outer tolerances are deliberately excluded from that fingerprint.

A missing, malformed, uncertified, non-finite, cross-shift-failed, or hard-physical-
failed point is never converted into a zero integrand. The adaptive result becomes
`unresolved`.

## Radial estimator

For every active panel `[a,b]`, the controller evaluates:

```text
coarse = fixed-order Gauss-Legendre rule on [a,b]
fine   = same rule on [a,(a+b)/2] + same rule on [(a+b)/2,b]
error  = abs(fine - coarse)
```

The accepted leaf estimate is `fine`. If a panel is selected for refinement, its
two child rules become the parent rules of the next round and their exact q nodes
are reused through the certified-point cache.

Errors are not checked only after summing all channels. The controller separately
requires convergence for every requested pairing and every Matsubara contribution:

```text
sum_panel_errors_J_m2 <= max(radial_atol_J_m2,
                              radial_rtol * abs(contribution_J_m2))
```

The next refinement wave chooses the highest normalized-error panels, subject to:

- maximum refinement rounds;
- maximum panel depth;
- maximum microscopic q-node budget;
- maximum panels refined per round.

Exhausting any limit before convergence returns `unresolved` with an explicit
termination reason.

## Result and audit surface

The result records:

- the finite partial free energy and each Matsubara contribution;
- the accepted outer-Q integral for every channel;
- estimated radial errors and channel tolerances;
- final leaf panels, depths and local scores;
- refinement round count and unique microscopic q-node count;
- cache hits, new point evaluations and certification batch count;
- unresolved microscopic points and the termination reason.

The fixed controller remains the regression authority until this adaptive path is
separately qualified against analytic integrands, dense fixed grids, the reviewed
`spm, n=0,1` reference, real microscopic runs, restart behavior and deterministic
serial/process execution.
