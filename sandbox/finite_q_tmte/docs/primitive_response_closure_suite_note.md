# Primitive response closure suite note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_primitive_response_closure_suite.py
```

It is not a production convention proposal.

## Motivation

The response-level decomposition showed that the fixed inverse-Green primitive Ward vector leaves a residual dominated by the electromagnetic sector, while the collective sector is almost closed. The next task is to determine whether the remaining EM residual is consistent with:

```text
1. a contact/direct normalization error;
2. a mixed K_etaS/K_Seta normalization or phase error;
3. a Schur correction normalization error;
4. a simple sign/phase convention error;
5. or a deeper endpoint/bubble-routing mismatch.
```

This suite runs all of these diagnostic checks at once.

## Fixed Ward vector

By default, the suite uses the matrix-inferred Matsubara primitive vector:

```text
observable side:
  u_L = (+i xi, +q, 0)
  W_L = (0, -2i Delta0)

source side:
  u_R = (-i xi, +q, 0)
  W_R = (0, -2i Delta0)
```

This vector is fixed by the inverse-Green matrix-level Ward block and is not re-fit by this audit.

## Sectors analyzed

The suite analyzes three three-term balances on both left and right sides.

EM balance:

```text
bubble + contact + mixed
```

where the terms are

```text
u K_SS_bubble
u K_SS_contact
W K_etaS
```

or the corresponding right-side contraction.

Schur balance:

```text
bubble + contact - schur_correction
```

Collective balance:

```text
mixed + K_etaeta_bubble + K_etaeta_counterterm
```

## Fits and grids

For each sector, the suite reports:

```text
1. current total residual;
2. best real and complex one-scale fit of each term;
3. best real and complex two-scale fit for the two non-fixed terms;
4. a sign/phase grid using coefficients {+1, -1, +i, -i};
5. cancellation overlaps between dominant terms.
```

## Interpretation

If a one-scale contact fit gives a coefficient close to `1` and does not improve much, the contact normalization is probably not the main issue.

If a one-scale contact fit gives a coefficient far from `1` and strongly improves the residual, contact/direct normalization is a strong suspect.

If the best sign/phase grid improves strongly with a coefficient such as `-1`, `+i`, or `-i`, the issue may be a simple sign or phase convention.

If the best complex two-scale fit can nearly close the sector but requires unphysical coefficients far from unit magnitude, the issue is probably not a simple convention but a mismatch between response block definitions.

If none of the one-scale or sign/phase fits can substantially reduce the residual, the remaining mismatch is likely in endpoint routing, band-basis bubble assembly, or missing direct terms rather than a scalar normalization.

## Non-goal

This audit does not accept any convention. It only prioritizes the next analytic or code-level investigation.
