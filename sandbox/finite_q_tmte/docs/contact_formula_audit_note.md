# Contact formula audit note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_contact_formula_audit.py
```

It is not a production convention proposal.

## Analytic input

The finite-q Matsubara primitive Ward vector is fixed by the inverse-Green matrix-level Ward identity:

```text
observable side:
  u_L = (+i xi, +q, 0)
  W_L = (0, -2i Delta0)

source side:
  u_R = (-i xi, +q, 0)
  W_R = (0, -2i Delta0)
```

The response kernel is split as

```text
K_SS = K_SS_bubble + K_SS_contact
```

The response-level Ward identity requires

```text
u_L K_SS + W_L K_etaS = 0
K_SS u_R + K_Seta W_R = 0
```

Therefore the contact contraction required by the Ward identity is

```text
left_contact_required  = - u_L K_SS_bubble - W_L K_etaS
right_contact_required = - K_SS_bubble u_R - K_Seta W_R
```

The implemented contact contractions are

```text
left_contact_current  = u_L K_SS_contact
right_contact_current = K_SS_contact u_R
```

The audit compares these current and required contact contractions directly.

## What the scalar means

The audit reports

```text
alpha_required_over_current
```

which minimizes

```text
|| contact_required - alpha contact_current ||
```

This scalar is diagnostic only. It must not be used as a production contact coefficient.

If `alpha` is close to 1 and the projection residual is small, the implemented contact contraction matches the Ward-required contact contraction in that direction.

If `alpha` is not close to 1 but the projection residual is small, the implemented contact contraction has the right direction but wrong magnitude in the tested Ward direction.

If the projection residual is not small or component-wise ratios disagree, the scalar is only a projection artifact. The issue is then more likely in the contact tensor, primitive projection, or endpoint routing.

## Why this audit follows the closure suite

The closure suite found that the EM residual can be removed by a scalar contact coefficient near 0.8268 in the dwave n=1 q=0.02 shifted-mesh run. This does not mean the correct production coefficient is 0.8268.

It means the current EM Ward residual is almost parallel to the current contact contraction. The next question is whether the Ward-required contact contraction is simply a scalar multiple of the implemented contact contraction, or whether the component-level structure differs.

## Expected interpretation

If the left and right scalar ratios agree, source/observable asymmetry is unlikely to be the main problem.

If the component-wise ratios agree across A0/L/T, the mismatch may be a scalar normalization or finite-q contact-definition issue.

If the component-wise ratios disagree, do not apply a scalar correction. Inspect the primitive tensor construction and source/observable projection.

If A0 components of the contact contraction are nonzero in a way not induced by the Ward contraction, inspect whether contact is being constructed in target G/TM/TE basis rather than primitive A0/L/T basis.

## Non-goal

This audit does not modify contact terms and does not accept conventions. It only compares implemented contact contractions to Ward-required contact contractions.
