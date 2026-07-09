# Schur effective translation RHS audit note

This diagnostic checks how the primitive finite-q translation/contact RHS transforms after collective Schur elimination.

Script:

```text
sandbox/finite_q_tmte/scripts/debug_schur_effective_translation_rhs_audit.py
```

## Primitive input identity

The primitive extended audit established, over no-shift/shifted2/shifted5 scans, that

```text
u K_SS + W K_etaS = R_S
```

with

```text
R_S = translation_forward + qM_mid
```

for the selected matrix-inferred Matsubara Ward candidate.

## Schur-effective identity

The Schur effective kernel is

```text
K_eff = K_SS - K_Seta K_etaeta^{-1} K_etaS
```

The collective-channel Ward vector is measured as

```text
C_eta = u K_Seta + W K_etaeta
```

on the left side, and analogously

```text
C_eta^R = K_etaS u + K_etaeta W
```

on the right side.

Then the Schur-projected Ward identity is

```text
u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS
```

and on the right side

```text
K_eff u = R_S - K_Seta K_etaeta^{-1} C_eta^R
```

## Output

The audit reports:

```text
S-channel residual:       u K_SS + W K_etaS - R_S
eta-channel C_eta:        u K_Seta + W K_etaeta
eta projection:           C_eta K_etaeta^{-1} K_etaS
effective direct:         u K_eff
effective predicted RHS:  R_S - eta projection
effective residual:       u K_eff - predicted RHS
```

and the analogous right-side quantities.

## Interpretation

If both `S-channel residual` and `effective residual` are small, the Schur-effective residual is explained by Schur projection of the primitive translation/contact RHS and the measured collective-channel vector.

This is diagnostic-only. It does not modify production kernels and does not validate Casimir input.
