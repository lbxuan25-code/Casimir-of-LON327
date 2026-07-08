# Finite-q Matsubara Ward convention note

This note records the current diagnostic status of the finite-q Ward residual investigation. It is intentionally not a production convention proposal.

## Current localized blocker

The finite-q TM/TE diagnostics have localized the robust Ward residual to the phase-like collective channel mixed with the TM source. Further primitive decomposition shows that the problematic part is dominated by the density-phase primitive mixed block rather than the longitudinal current-phase primitive mixed block.

Observed diagnostic fingerprint at the current representative point:

```text
K_etaS[phase_eta2, A0]  is predominantly real
K_etaS[phase_eta2, L]   is predominantly imaginary
```

The current target basis uses real coefficients:

```text
G  =  xi * A0 + q * L
TM = -q  * A0 + xi * L
TE = T
```

This mixes a real density-phase primitive block with an imaginary current-phase primitive block. The resulting target-basis `phase_eta2-TM` block generates the robust imaginary `G-TM` Ward residual after the collective Schur completion.

## Important rule

Do not change production conventions based on residual minimization alone.

A debug-only basis scan may identify a candidate fingerprint, but it is not an accepted convention. Any production convention change must be backed by an analytic Matsubara Ward derivation explaining:

1. why the current convention is wrong;
2. why the proposed convention is correct;
3. how the same convention acts on both source and observable rows/columns;
4. how the order-parameter phase variable `eta2` is normalized;
5. how the result follows from the imaginary-time BdG/Nambu action and gauge transformation.

## Required analytic questions

Before any production change, derive the finite-q imaginary-time Ward generator from the action-level convention used by this repository:

```text
Psi -> exp(i tau_3 chi) Psi
Delta -> exp(2 i chi) Delta
```

Resolve the following explicitly:

- Does the scalar source enter the Euclidean BdG kernel as `A0`, `i A0`, or with another sign?
- Does the Matsubara frequency component in the Ward generator appear as `xi`, `i xi`, `-i xi`, or another convention-dependent factor?
- Does the longitudinal current component carry an additional `i` relative to the density component?
- Is the phase variable defined as `delta Delta = i eta2 phi`, `eta2 = Delta0 theta`, or with a different normalization?
- Should the left/source and right/observable Ward vectors use the same sign or opposite signs under the existing `add_bubble` convention?

## Role of the scan

The script `sandbox/finite_q_tmte/scripts/debug_ward_basis_convention_scan.py` exists only to fingerprint possible convention issues. It reconstructs primitive blocks from the current real target basis, applies diagnostic candidate target transforms, and reports residuals.

It must not mark a winner. It must not set `valid_for_casimir_input=True`. It must not be used as a production fix.

A candidate convention can only be considered after:

1. the analytic derivation uniquely predicts it;
2. the debug scan is consistent with it;
3. q-, nk-, and Matsubara-index sweeps verify stable Ward improvement;
4. the change is propagated coherently through target vertices, component basis, contact projection, and documentation.
