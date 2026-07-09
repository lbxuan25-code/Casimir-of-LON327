# Finite-q Ward closure status

This note summarizes the current state of the finite-q Ward residual work after the RHS-aware primitive and Schur-effective diagnostics.

Status: Ward-specific work is complete at the sandbox diagnostic level.  Physical response convergence is not complete.  Production/Casimir readiness is not achieved.

---

## 1. What is now closed

The old zero-RHS finite-q Ward criterion is invalid.  The finite-q Ward identity has a nonzero RHS from translation/equal-time and Peierls contact terms.

The primitive extended identity is

```text
u K_SS + W K_etaS = R_S,
R_S = translation_forward + qM_mid.
```

The Schur-effective identity is

```text
u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS,
C_eta = u K_Seta + W K_etaeta.
```

Right-side identities use the analogous source-side contractions:

```text
K_SS u + K_Seta W = R_S^R,
K_eff u = R_S^R - K_Seta K_etaeta^{-1} C_eta^R,
C_eta^R = K_etaS u + K_etaeta W.
```

Both left and right diagnostics close numerically.

---

## 2. Diagnostic evidence

Primitive extended robustness:

```text
no-shift:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=9,13
  36/36 passed.

shifted2:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=9,13
  36/36 passed.

shifted5:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=13,17
  36/36 passed.
```

Schur-effective diagnostic:

```text
shifted5 main point, dwave n=1 q=0.02 nk=13:
  effective residual/reference about 1.1e-13.

shifted5 main point, spm n=1 q=0.02 nk=13:
  effective residual/reference about 1.4e-12.

no-shift light scan:
  spm, dwave; n=1,2,3; q=0.01,0.02,0.04; nk=9,13
  effective residual/reference about 1e-15 to 2.25e-13.
```

RHS-aware validation staging:

```text
dwave, shifted5, n=1, q=0.02, nk=13:
  rhs_aware_ward_closed = True
  max_S_res/rhs = 1.08053316e-13
  max_eff_res/ref = 1.06907163e-13
  legacy_zero_rhs/Keff = 3.02479763e-03

spm, shifted5, n=1, q=0.02, nk=13:
  rhs_aware_ward_closed = True
  max_S_res/rhs = 1.40334963e-12
  max_eff_res/ref = 1.41435872e-12
  legacy_zero_rhs/Keff = 7.18476027e-05
```

These results show that the residual previously seen in `K_eff` is a Schur-projected finite-q RHS, not an unexplained Ward failure.

---

## 3. Main diagnostic files

Primitive RHS-aware audit:

```text
sandbox/finite_q_tmte/scripts/debug_primitive_extended_translation_collective_audit.py
sandbox/finite_q_tmte/tmte/pipeline/primitive_extended_translation_collective_audit.py
```

Primitive robustness scan:

```text
sandbox/finite_q_tmte/scripts/debug_primitive_extended_translation_collective_robustness_scan.py
sandbox/finite_q_tmte/tmte/pipeline/primitive_extended_translation_collective_robustness_scan.py
```

Schur-effective RHS-aware audit:

```text
sandbox/finite_q_tmte/scripts/debug_schur_effective_translation_rhs_audit.py
sandbox/finite_q_tmte/tmte/pipeline/schur_effective_translation_rhs_audit.py
```

Sandbox RHS-aware validation staging:

```text
sandbox/finite_q_tmte/scripts/debug_rhs_aware_finite_q_validation.py
sandbox/finite_q_tmte/tmte/pipeline/rhs_aware_finite_q_validation.py
sandbox/finite_q_tmte/scripts/debug_rhs_aware_convergence_scan.py
sandbox/finite_q_tmte/tmte/pipeline/rhs_aware_convergence_scan.py
```

Analytic and handoff notes:

```text
sandbox/finite_q_tmte/docs/finite_q_bdg_schur_ward_derivation.md
sandbox/finite_q_tmte/docs/rhs_aware_finite_q_validation_note.md
sandbox/finite_q_tmte/docs/finite_q_ward_final_handoff.md
```

---

## 4. Interpretation update

Old interpretation:

```text
K_eff Ward residual means finite-q BdG response has not closed.
```

Updated interpretation:

```text
The zero-RHS target was wrong at finite q.  The residual equals the finite-q translation/contact RHS after collective Schur projection.
```

The contact-scaling numbers such as the historical `~0.8268` are therefore not physical correction coefficients.  They are artifacts of forcing a zero-RHS closure target onto a finite-q identity with a nonzero RHS.

---

## 5. Sandbox validation staging now available

The sandbox now has a production-style but diagnostic-only validation summary:

```text
rhs_aware_ward_closed
primitive_s_channel_closed
schur_effective_closed
condition_ok
legacy_zero_rhs_check.status = invalid_target_at_finite_q
valid_for_casimir_input = False
```

It also has a norm-level convergence scan that compares adjacent `nk` values and shift modes for:

```text
K_eff norm
R_eff norm
eta projection / R_S
K_etaeta condition number
```

The latest convergence diagnostics confirm the boundary:

```text
no-shift light scan:
  num_rows = 36
  num_rhs_aware_closed = 36
  all_rhs_aware_closed = True
  max_nk_relative_change_K_eff_norm = 2.40427139e-01
  max_nk_relative_change_R_eff_norm = 9.13985107e-01
  max_nk_relative_change_eta_projection_over_rhs_s = 9.53879376e-01

shift comparison at n=1, q=0.02, nk=13:
  num_rows = 6
  num_rhs_aware_closed = 6
  all_rhs_aware_closed = True
  max_shift_relative_change_K_eff_norm = 3.52423884e-01
  max_shift_relative_change_R_eff_norm = 9.93591687e-01
  max_shift_relative_change_eta_projection_over_rhs_s = 9.68565728e-01
```

Thus Ward closure is complete, while physical response convergence remains a separate unsolved problem.

---

## 6. Still not Casimir-ready

The current status remains

```text
valid_for_casimir_input: False
```

because the main production validator and scan runner have not been replaced, and the convergence scan is only a norm-level diagnostic.  It does not yet define a physical Casimir error budget.

A production-facing validation should check and report at least:

```text
RHS-aware primitive residual
RHS-aware Schur-effective residual
|R_eff| / |K_eff|
|C_eta K_etaeta^{-1} K_etaS| / |R_S|
condition number of K_etaeta
nk convergence
shift convergence
```

The production runner should not use fitted contact scales or any zero-RHS finite-q Ward pass/fail criterion.

---

## 7. Final Ward handoff

Ward-specific work should now stop at the sandbox diagnostic level.  The next stage is not to keep searching for Ward residual fixes; it is to solve physical response convergence and define the future Casimir-consumed response combination.

Recommended next work outside the Ward closure scope:

```text
1. Decide which K_eff components or projected combinations will be consumed by the future Casimir path.
2. Extend convergence diagnostics from norms to those final response combinations.
3. Study nk and shift convergence at higher nk and practical shift modes.
4. Define an error budget before reconsidering valid_for_casimir_input.
```

Until those tasks are complete, the finite-q path remains diagnostic-only.
