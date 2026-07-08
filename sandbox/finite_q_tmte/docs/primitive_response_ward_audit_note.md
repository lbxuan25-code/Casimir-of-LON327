# Primitive response Ward audit note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_primitive_response_ward_audit.py
```

It is not a production convention proposal.

## Motivation

The inverse-Green matrix audit found machine-precision Nambu Ward identities at the single-particle matrix level. For Matsubara transfer `z_+-z_-=i xi`, the matrix-level fingerprints suggest asymmetric primitive Ward generators rather than the current real symmetric target basis.

The next question is whether this matrix-level identity survives the response-level Kubo assembly and collective Schur completion when tested directly in primitive `(A0, L, T)` variables.

## What the audit computes

The audit reconstructs primitive response blocks from the current baseline target blocks:

```text
K_SS primitive
K_Seta primitive
K_etaS primitive
K_etaeta
```

Then it evaluates Ward contractions directly in primitive space:

```text
left bare EM:        u_L K_SS + W_L K_etaS
left collective:     u_L K_Seta + W_L K_etaeta
right bare EM:       K_SS u_R + K_Seta W_R
right collective:    K_etaS u_R + K_etaeta W_R
left Schur:          u_L K_eff
right Schur:         K_eff u_R
```

The main candidate is inferred from the inverse-Green audit:

```text
u_L = (+i xi, +q, 0)
W_L = (0, -2i Delta0)

u_R = (-i xi, +q, 0)
W_R = (0, -2i Delta0)
```

The audit also reports equivalent right-side overall sign flips and current real-target negative controls.

## Interpretation

If the matrix-inferred primitive candidate strongly improves both bare extended Ward identities and Schur effective residuals, then the main blocker is likely the current real symmetric `G/TM` target basis.

If the matrix-inferred primitive candidate does not improve response-level residuals, then the remaining mismatch is likely in the mapping from matrix Ward identities to finite-q Kubo response blocks: band-basis row/column ordering, endpoint routing, contact/direct term, or collective counterterm mapping.

In either case, this audit does not accept a convention. It only determines whether the inverse-Green Ward generator propagates to the response-level kernel.
