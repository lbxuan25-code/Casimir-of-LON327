# Finite-q BdG Ward identity and Schur-effective RHS

This note freezes the analytic interpretation behind the finite-q Ward diagnostics.  It is diagnostic-only and does not make the pipeline Casimir-ready.

## 1. Conventions

Finite-q routing:

```text
k_minus = k - q/2
k_plus  = k + q/2
```

Primitive external order:

```text
S = (A0, L, T)
```

Collective order:

```text
eta = (amplitude, phase_eta2)
```

The closed matrix-inferred Matsubara candidate uses

```text
left:
  u_L = (+i xi, +q, 0)
  W_L = (0, -2 i Delta0)

right:
  u_R = (-i xi, +q, 0)
  W_R = (0, -2 i Delta0)
```

Here `xi = 2 pi n k_B T`.  The right-side Matsubara sign is a source-side convention.

## 2. Matrix-level Ward identity

The left BdG matrix identity has the form

```text
G_+^{-1} tau3 - tau3 G_-^{-1}
  = sum_a u_L[a] Gamma_a + sum_alpha W_L[alpha] Lambda_alpha.
```

`Gamma_a` are primitive electromagnetic vertices and `Lambda_alpha` are collective vertices.  The phase collective vertex is required because a gauge transformation rotates the superconducting order parameter.

The right identity is the corresponding source-side identity:

```text
tau3 G_+^{-1} - G_-^{-1} tau3
  = sum_b Gamma_b u_R[b] + sum_beta Lambda_beta W_R[beta].
```

These matrix identities are the starting point.  The response-level RHS is not guessed from residuals.

## 3. Bubble Ward contraction

For a bubble block,

```text
K_AB^bub ~ Tr[Gamma_A G_+ Gamma_B G_-].
```

Contracting the left observable index gives

```text
u_L K_SS^bub + W_L K_etaS^bub
  ~ Tr[(u_L Gamma_S + W_L Lambda_eta) G_+ Gamma_S G_-].
```

Using the matrix Ward identity, the inverse Green functions collapse the adjacent propagators.  The finite-temperature sum leaves an equal-time commutator / one-point term.  At finite q this term is not zero.  The diagnostics call it

```text
translation_forward.
```

Thus

```text
u_L K_SS^bub + W_L K_etaS^bub = translation_forward.
```

The right-side bubble identity is analogous.

## 4. Peierls contact and qM_mid

The full electromagnetic block contains a Peierls contact term:

```text
K_SS = K_SS^bub + K_SS^contact.
```

The Ward contraction of the spatial contact block gives the finite-q diamagnetic/contact contribution.  In the current q along x primitive longitudinal channel this diagnostic vector is

```text
qM_mid.
```

Therefore the full primitive external-channel Ward identity is

```text
u_L K_SS + W_L K_etaS = R_S,
R_S = translation_forward + qM_mid.
```

On the right side,

```text
K_SS u_R + K_Seta W_R = R_S^R.
```

In the current diagnostic convention `R_S^R` is represented by the same primitive ordered `translation_plus_qM` vector.  This relies on the fixed source/observable sign convention.

## 5. Collective channel

The collective source column has its own Ward contraction:

```text
C_eta := u_L K_Seta + W_L K_etaeta.
```

Right side:

```text
C_eta^R := K_etaS u_R + K_etaeta W_R.
```

One must not set this object to zero by assumption.  It contains the collective-column equal-time term and the variation of the collective counterterm / gap-action sector.  The Schur audit measures `C_eta` directly.

## 6. Schur-effective identity

The Schur-effective external kernel is

```text
K_eff = K_SS - K_Seta K_etaeta^{-1} K_etaS.
```

Starting from

```text
u_L K_SS + W_L K_etaS = R_S,
u_L K_Seta + W_L K_etaeta = C_eta,
```

we have

```text
u_L K_eff
  = u_L K_SS - u_L K_Seta K_etaeta^{-1} K_etaS
  = R_S - C_eta K_etaeta^{-1} K_etaS.
```

Right side:

```text
K_eff u_R = R_S^R - K_Seta K_etaeta^{-1} C_eta^R.
```

This is block algebra following from the primitive identities.  It is not a fitted ansatz.

## 7. Diagnostic mapping

Primitive extended audit:

```text
u K_SS + W K_etaS = translation_forward + qM_mid.
```

Because the audit reports `missing_to_close = -(u K_SS + W K_etaS)`, the expected top candidate is

```text
minus_translation_plus_qM.
```

Schur-effective audit:

```text
u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS.
```

It reports the S-channel residual, `C_eta`, the Schur projection, direct `K_eff` contraction, predicted RHS, residual, and `K_etaeta` condition number.

## 8. Numerical status

RHS-aware diagnostics passed the following windows with residuals far below `1e-9`:

```text
Primitive extended no-shift:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=9,13; 36/36 passed.

Primitive extended shifted2:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=9,13; 36/36 passed.

Primitive extended shifted5:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=13,17; 36/36 passed.

Schur-effective no-shift light scan:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=9,13;
  effective residual/reference about 1e-15 to 2.25e-13.
```

Representative shifted5 main points:

```text
dwave, n=1, q=0.02, nk=13:
  effective residual/reference about 1.1e-13
  eta projection / R_S about 1.1e-2

spm, n=1, q=0.02, nk=13:
  effective residual/reference about 1.4e-12
  eta projection / R_S about 1.25
```

The `dwave` and `spm` difference is a difference in Schur-projected RHS size, not a Ward-closure failure.

## 9. Production implications

The finite-q validator must not use

```text
u K_eff ~= 0.
```

The RHS-aware check is

```text
u K_eff - (R_S - C_eta K_etaeta^{-1} K_etaS) ~= 0.
```

A production-facing diagnostic should also report:

```text
RHS-aware Schur residual
|R_eff| / |K_eff|
|C_eta K_etaeta^{-1} K_etaS| / |R_S|
condition number of K_etaeta
nk/shift convergence of physical responses
```

All results in this note remain `valid_for_casimir_input = False` until a production validator and convergence policy are implemented.
