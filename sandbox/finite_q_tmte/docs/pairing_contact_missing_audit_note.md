# Pairing contact missing audit note

This note describes the diagnostic purpose of

```text
sandbox/finite_q_tmte/scripts/debug_pairing_contact_missing_audit.py
```

It is not a production convention proposal.

## Motivation

The contact formula audit found that, for the dwave n=1 q=0.02 shifted-mesh run, the Ward-required contact contraction is almost exactly parallel to the implemented contact contraction, but smaller by a scalar factor near 0.8268.

This does not mean the correct production fix is to multiply contact by 0.8268.

The next question is whether this mismatch comes from:

```text
1. a normal-state Peierls contact problem;
2. a superconducting pairing-sector contact/direct contribution missing from K_SS_contact;
3. a momentum-dependent/bond-pairing gauge-covariance issue;
4. or a general finite-q discretization/endpoint issue.
```

## What the audit runs

The audit reuses `contact_formula_audit` for a small control grid:

```text
pairings = dwave, spm
delta0  = 0.00, 0.05, 0.10, 0.15 eV
```

For each point it reports:

```text
alpha_required_over_current
missing_fraction = 1 - Re(alpha)
parallelism_abs_overlap
projection_residual_over_required
left_right_alpha_abs_diff
```

It also fits, for each pairing,

```text
alpha_real_mean ~= intercept + slope * delta0_eV^2
```

This trend is diagnostic only.

## Interpretation

If `delta0=0` gives `alpha ~= 1`, the normal-state Peierls contact is likely consistent and the mismatch is tied to superconductivity.

If `spm` stays near `alpha ~= 1` while `dwave` deviates, momentum-dependent or bond-pairing gauge contact is a strong suspect.

If the deviation `1-alpha` scales with `delta0^2`, the missing contribution is likely pairing-amplitude controlled.

If both `spm` and `dwave` deviate similarly even at `delta0=0`, the issue is more likely a general normal-state contact or finite-q endpoint convention problem.

If parallelism is poor or component ratios disagree, the scalar alpha should not be interpreted as a contact normalization. The issue is then tensor/projection/routing rather than a scalar missing contribution.

## Non-goal

This audit does not derive or apply a corrected contact term. It only distinguishes which analytic path should be pursued next.
