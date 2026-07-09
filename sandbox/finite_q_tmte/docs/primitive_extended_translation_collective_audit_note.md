# Primitive extended translation collective audit note

This diagnostic extends the primitive EM translation-RHS audit by adding the mixed collective Ward term.

Script:

```text
sandbox/finite_q_tmte/scripts/debug_primitive_extended_translation_collective_audit.py
```

## Scope

The previous primitive EM translation audit checked only:

```text
u K_SS
```

This audit checks:

```text
u K_SS + W K_etaS
K_SS u + K_Seta W
```

for the selected primitive Ward candidate.

It still does not include full Schur closure through `K_etaeta`. Therefore it is not a Casimir-ready production validation.

## Purpose

The normal sector showed that finite-q zero-RHS Ward tests are incomplete because the response-level residual equals a translation/equal-time RHS.

The primitive EM-only BdG audit showed that a normal-like translation RHS explains much of the residual direction but leaves a non-negligible orthogonal remainder.

This audit asks whether including the mixed collective term `W K_etaS` makes the remaining primitive residual match the same translation/equal-time RHS.

## Interpretation

If a translation-family candidate ranks first with unit coefficient and small direct mismatch, then the primitive BdG residual is mostly a finite-q translation RHS after including the phase-mixed collective term.

If translation only matches directionally after a fitted scalar, the result is not yet a formula.

If no translation-family candidate is close, the remaining residual likely requires BdG/pairing/collective equal-time structure beyond the normal-like translation RHS.

## Non-goal

This audit does not modify production response code and does not validate Casimir input.
