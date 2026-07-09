# Finite-q Ward final handoff

This is the final handoff for the sandbox finite-q Ward residual work.  The Ward-specific work is considered complete at the sandbox diagnostic level.

Main boundary:

```text
Ward closure problem: closed in sandbox diagnostics
Physical response convergence: not closed; belongs to the next work stage
Main validation flow: unchanged
valid_for_casimir_input: False
```

---

## 1. Final Ward identity

The finite-q Ward target is RHS-aware.  The zero-RHS target is invalid at finite q.

Primitive extended identity:

```text
u K_SS + W K_etaS = R_S,
R_S = translation_forward + qM_mid.
```

Schur-effective identity:

```text
u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS,
C_eta = u K_Seta + W K_etaeta.
```

Right-side identities are the analogous source-side contractions:

```text
K_SS u + K_Seta W = R_S^R,
K_eff u = R_S^R - K_Seta K_etaeta^{-1} C_eta^R,
C_eta^R = K_etaS u + K_etaeta W.
```

These identities are documented in:

```text
sandbox/finite_q_tmte/docs/finite_q_bdg_schur_ward_derivation.md
```

---

## 2. Final sandbox validation status

The staged RHS-aware validation summary is implemented in:

```text
sandbox/finite_q_tmte/scripts/debug_rhs_aware_finite_q_validation.py
sandbox/finite_q_tmte/tmte/pipeline/rhs_aware_finite_q_validation.py
```

Representative shifted5 results at `n=1`, `q=0.02`, `nk=13`:

```text
dwave:
  rhs_aware_ward_closed: True
  primitive_s_channel_closed: True
  schur_effective_closed: True
  condition_ok: True
  max_S_res/rhs  = 1.08053316e-13
  max_eff_res/ref = 1.06907163e-13
  max_eta_proj/rhs = 1.09518276e-02
  legacy_zero_rhs/Keff = 3.02479763e-03
  cond(K_etaeta) = 3.50152978e+01

spm:
  rhs_aware_ward_closed: True
  primitive_s_channel_closed: True
  schur_effective_closed: True
  condition_ok: True
  max_S_res/rhs  = 1.40334963e-12
  max_eff_res/ref = 1.41435872e-12
  max_eta_proj/rhs = 1.24859420e+00
  legacy_zero_rhs/Keff = 7.18476027e-05
  cond(K_etaeta) = 5.23838848e+01
```

Interpretation:

```text
The RHS-aware Ward identity closes.
The old zero-RHS residual is reported only as an invalid finite-q legacy target.
```

---

## 3. Final convergence diagnostic status

The norm-level convergence scan is implemented in:

```text
sandbox/finite_q_tmte/scripts/debug_rhs_aware_convergence_scan.py
sandbox/finite_q_tmte/tmte/pipeline/rhs_aware_convergence_scan.py
```

The scan confirms Ward closure but shows that physical response convergence is not solved.

No-shift light scan:

```text
pairings: spm, dwave
n: 1, 2, 3
q: 0.01, 0.02, 0.04
nk: 9, 13
shift: noshift

num_rows = 36
num_rhs_aware_closed = 36
all_rhs_aware_closed = True
max_S_res/rhs = 5.58114143e-13
max_eff_res/ref = 2.24545157e-13
max_eta_projection_over_rhs_s = 7.83990391e+00
max_legacy_zero_rhs/Keff = 2.50195049e-02
max_cond(K_etaeta) = 4.14282414e+02
max_nk_relative_change_K_eff_norm = 2.40427139e-01
max_nk_relative_change_R_eff_norm = 9.13985107e-01
max_nk_relative_change_eta_projection_over_rhs_s = 9.53879376e-01
```

Representative shift comparison at `n=1`, `q=0.02`, `nk=13`:

```text
num_rows = 6
num_rhs_aware_closed = 6
all_rhs_aware_closed = True
max_S_res/rhs = 3.68955737e-12
max_eff_res/ref = 1.41435872e-12
max_eta_projection_over_rhs_s = 3.02162919e+00
max_legacy_zero_rhs/Keff = 1.18247249e-02
max_cond(K_etaeta) = 5.23838848e+01
max_shift_relative_change_K_eff_norm = 3.52423884e-01
max_shift_relative_change_R_eff_norm = 9.93591687e-01
max_shift_relative_change_eta_projection_over_rhs_s = 9.68565728e-01
```

Interpretation:

```text
Ward closure is robust.
K_eff, R_eff, and eta-projection norms are not yet converged with respect to nk/shift.
This is a numerical/physical response convergence problem, not a Ward residual problem.
```

---

## 4. What not to do next

Do not continue fitting contact scales or scalar alpha values to force zero-RHS closure.

Do not treat `u K_eff ~= 0` as a finite-q Ward pass/fail criterion.

Do not set `valid_for_casimir_input = True` from RHS-aware Ward closure alone.

Do not move these sandbox diagnostics into the main validation flow until the sandbox finite-q calculation basis replaces the old main-flow basis.

---

## 5. Next work stage

The next stage is not Ward closure.  It is physical response convergence and definition of the final Casimir-consumed response combination.

Recommended next tasks:

```text
1. Decide which K_eff components or projected combinations will be consumed by the future Casimir path.
2. Extend convergence diagnostics from norms to those final response combinations.
3. Study nk and shift convergence at higher nk and practical shift modes.
4. Define an error budget before reconsidering valid_for_casimir_input.
```

Until those tasks are complete, the finite-q path remains diagnostic-only.
