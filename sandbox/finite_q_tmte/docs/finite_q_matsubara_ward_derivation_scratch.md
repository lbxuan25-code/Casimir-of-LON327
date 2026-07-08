# Finite-q Matsubara Ward derivation scratch

This scratch note describes the analytic task that must follow the diagnostic vertex convention audit. It is not a production convention proposal.

## Goal

Derive the finite-q imaginary-time Ward generator from the same BdG/Nambu and source conventions that the code uses, then map the result onto the repository's primitive vertices:

```text
Gamma_A0
Gamma_L
Gamma_eta2
```

The derivation must determine the coefficients multiplying these vertices in the Ward generator. In particular, it must decide whether the time component is proportional to `xi`, `i xi`, or `-i xi`, and whether the phase coefficient is proportional to `2 Delta0`, `2 i Delta0`, or their negatives.

## Minimal action-level starting point

Start from an imaginary-time Nambu action of the form

```text
S = int d tau sum_k Psi^dagger [partial_tau + H_BdG(A, Delta)] Psi + S_HS[Delta]
```

Use the local gauge transformation

```text
Psi  -> exp(i tau_3 chi) Psi
Delta -> exp(2 i chi) Delta
```

The derivation must keep track of the repository's conventions for:

- the sign and normalization of the density vertex;
- the sign and normalization of the Peierls current vertex;
- the definition of the phase collective channel `eta2`;
- the left/right source versus observable vertex convention used by `add_bubble`.

## Current diagnostic fingerprint

The existing diagnostics show that the robust G-TM Ward residual is dominated by the phase-density primitive mixed block:

```text
K_eta2,A0 is predominantly real
K_eta2,L  is predominantly imaginary
```

This is not by itself a fix. It is a constraint that the analytic derivation must explain.

## Required derivation checkpoints

1. Derive the temporal Ward term produced by `Psi^dagger partial_tau Psi` under `Psi -> exp(i tau_3 chi) Psi`.
2. Fourier transform the temporal term using the same Matsubara convention assumed by the response code.
3. Derive the spatial Ward term from the Peierls substitution used by the current vertex.
4. Derive the phase collective term from `Delta -> exp(2 i chi) Delta` and map it to the code's `phase_eta2` vertex.
5. Determine whether the response-level left and right Ward vectors must have the same sign or opposite signs under the current `add_bubble` convention.
6. Only after these steps, compare the predicted generator against the diagnostic matrix-level audit and response-level Ward residuals.

## Non-goal

Do not infer a production convention from whichever diagnostic candidate minimizes a residual. A convention is acceptable only if the analytic derivation predicts it and subsequent q-, nk-, and Matsubara-index sweeps verify it.
