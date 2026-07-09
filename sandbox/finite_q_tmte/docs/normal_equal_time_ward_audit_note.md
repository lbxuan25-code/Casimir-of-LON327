# Normal equal-time Ward audit note

This diagnostic tests whether the normal response-level Ward residual can be identified with an equal-time or finite-q translation term.

Script:

```text
sandbox/finite_q_tmte/scripts/debug_normal_equal_time_ward_audit.py
```

It is not a production convention proposal.

## Motivation

The normal Peierls vertex identity closes at machine precision in absolute error, but normal bubble plus contact does not close at the response level. A scan over simple response conventions did not find a closure.

Therefore the next question is not which sign to flip. The next question is which equal-time/contact-like term is missing from the response-level Ward identity.

## What the audit computes

The audit uses the baseline normal convention:

```text
observable = (rho, -Vx, -Vy)
source     = (rho, +Vx, +Vy)
contact    = -<M_ij>
Ward left  = (i xi, +qx, +qy) K
Ward right = K (i xi, -qx, -qy)^T
```

It computes:

```text
B = Ward contraction of K_bubble
C = Ward contraction of K_contact
R = B + C
missing_to_close = -R
```

Then it compares `missing_to_close` against equal-time candidates:

```text
actual_equal_time_forward
actual_equal_time_direct
shifted vector difference deltaV(k+q/2)-deltaV(k-q/2)
translation_error = actual_equal_time - shifted_delta_v
qM_mid
shifted_delta_v + qM_mid
```

Both signs of each candidate are included in the ranking.

## How to read the result

Look at:

```text
ranked equal-time candidates against left missing_to_close
```

If one candidate has `diff/missing << 1` without relying only on a fitted scalar, that candidate is likely the missing Ward term.

If a candidate only matches after an arbitrary scalar `fit_alpha`, then it is a directional fingerprint but not yet a formula.

If no candidate is close, the finite-q lattice Kubo Ward identity must be rederived more carefully.

## Non-goal

This audit does not modify the response kernel. It only localizes which equal-time structure should be analytically derived next.
