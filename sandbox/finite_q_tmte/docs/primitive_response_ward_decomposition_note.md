# Primitive response Ward decomposition note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_primitive_response_ward_decomposition.py
```

It is not a production convention proposal.

## Motivation

The inverse-Green matrix-level Ward block is now fixed:

```text
observable side:
  u_L = (+i xi, +q, 0)
  W_L = (0, -2i Delta0)

source side:
  u_R = (-i xi, +q, 0)
  W_R = (0, -2i Delta0)
```

The primitive response-level Ward audit showed that this vector improves the response residual relative to the old real target-like vector, but does not close the response Ward identity. Therefore the remaining residual must be decomposed rather than fitted by new Ward vectors.

## What the decomposition reports

For the selected primitive Ward vector, the script reports the following terms.

Left EM residual:

```text
u_L K_SS_bubble
u_L K_SS_contact
W_L K_etaS
total
```

Left collective residual:

```text
u_L K_Seta
W_L K_etaeta_bubble
W_L K_etaeta_counterterm
total
```

Right EM residual:

```text
K_SS_bubble u_R
K_SS_contact u_R
K_Seta W_R
total
```

Right collective residual:

```text
K_etaS u_R
K_etaeta_bubble W_R
K_etaeta_counterterm W_R
total
```

Schur effective residual:

```text
u_L K_SS_bubble
u_L K_SS_contact
- u_L Schur_correction
u_L K_eff total

K_SS_bubble u_R
K_SS_contact u_R
- Schur_correction u_R
K_eff u_R total
```

## Interpretation

If the residual is dominated by EM terms, focus on

```text
K_SS_bubble
K_SS_contact
K_etaS/K_Seta mixed blocks
finite-q endpoint routing
source/observable current sign convention
```

If the residual is dominated by collective terms, focus on

```text
K_etaeta_bubble
K_etaeta_counterterm
phase_eta2 normalization
Hubbard-Stratonovich counterterm convention
```

If the bare extended residual is small but the Schur effective residual is large, focus on the Schur completion and K_etaeta solve/counterterm structure.

If the Schur effective residual is small but bare extended residual is large, the collective Schur completion is doing the main Ward restoration and the remaining issue is likely in how the target physical basis is extracted.

## Non-goal

Do not select a production convention from this decomposition. It only localizes which response block prevents the fixed inverse-Green matrix Ward identity from closing at the response level.
