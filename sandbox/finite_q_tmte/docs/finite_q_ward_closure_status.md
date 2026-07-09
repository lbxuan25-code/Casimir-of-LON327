# Finite-q Ward closure status

This note summarizes the current state of the finite-q Ward residual work after the RHS-aware primitive and Schur-effective diagnostics.

Status: diagnostic closure achieved; sandbox RHS-aware validation staging implemented; production/Casimir readiness not yet achieved.

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

Analytic derivation note:

```text
sandbox/finite_q_tmte/docs/finite_q_bdg_schur_ward_derivation.md
sandbox/finite_q_tmte/docs/rhs_aware_finite_q_validation_note.md
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

This staging layer is intended for future migration into the main validation flow only after the sandbox finite-q calculation path replaces the old main-flow basis.

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

## 7. Recommended next work

1. Run the sandbox RHS-aware validation and convergence scan on representative finite-q points.
2. Decide which response combination is actually consumed by the future Casimir path.
3. Extend convergence diagnostics from norm-level summaries to the final Casimir-consumed response combination.
4. Keep `contact_scale` and scalar alpha projections diagnostic-only.
5. Only after the main validation flow is replaced and a convergence/error policy exists should `valid_for_casimir_input` be reconsidered.
