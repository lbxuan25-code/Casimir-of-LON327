# Primitive EM translation RHS audit note

This diagnostic checks whether the normal finite-q translation/equal-time RHS also explains the BdG primitive EM residual before collective Schur completion.

Script:

```text
sandbox/finite_q_tmte/scripts/debug_primitive_em_translation_rhs_audit.py
```

## Scope

This audit is primitive EM only:

```text
primitive order = [A0, Ax, Ay]
K_SS = K_SS_bubble + K_SS_contact
no collective Schur correction
no W_eta collective terms
```

It uses the matrix-inferred Matsubara Ward candidate by default:

```text
left_u  = [+i xi, +q, 0]
right_u = [-i xi, +q, 0]
```

## What it compares

The audit computes:

```text
left_total = left_u @ K_SS
left_missing_to_close = -left_total
```

Then it constructs BdG primitive analogues of the normal equal-time vectors:

```text
equal_forward
shifted_delta_v_mid
translation_forward = equal_forward - shifted_delta_v_mid
qM_mid
shifted_delta_v_mid + qM_mid
```

Both signs of each candidate are ranked against `left_missing_to_close`.

## Interpretation

If `minus_translation_forward` ranks first with unit coefficient and small direct mismatch, the primitive EM residual is primarily the same finite-q translation/equal-time RHS found in the normal sector.

If it only matches after an arbitrary fitted scalar, this is only a directional fingerprint.

If no translation-family candidate is close, the remaining BdG primitive EM residual cannot be explained by the normal-like translation RHS alone. In that case, the next suspects are superconducting phase/gauge completion and collective-sector Ward terms.

## Non-goal

This audit does not modify the response kernel and does not validate Casimir input. It is diagnostic-only.
