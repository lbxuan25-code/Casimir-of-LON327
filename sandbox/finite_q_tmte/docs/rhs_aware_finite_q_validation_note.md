# RHS-aware finite-q validation summary

This sandbox diagnostic wraps the Schur-effective RHS audit into a production-style validation summary without changing the main validation flow.

Scripts:

```text
sandbox/finite_q_tmte/scripts/debug_rhs_aware_finite_q_validation.py
sandbox/finite_q_tmte/scripts/debug_rhs_aware_convergence_scan.py
```

Pipelines:

```text
sandbox/finite_q_tmte/tmte/pipeline/rhs_aware_finite_q_validation.py
sandbox/finite_q_tmte/tmte/pipeline/rhs_aware_convergence_scan.py
```

Status: sandbox diagnostic only.  It always keeps `valid_for_casimir_input = False`.

---

## 1. Single-point RHS-aware validation

The single-point summary checks the RHS-aware Schur identity

```text
u K_eff = R_S - C_eta K_etaeta^{-1} K_etaS.
```

It reports:

```text
primitive S-channel residual / RHS
Schur-effective residual / reference
eta projection / R_S
legacy zero-RHS residual / K_eff
K_eff norm
K_etaeta condition number
```

The old zero-RHS check is not used as a pass/fail target.  It is reported as

```text
legacy_zero_rhs_check.status = invalid_target_at_finite_q
```

Example:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_rhs_aware_finite_q_validation.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --nk 13 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/dwave_n1_q002_nk13_rhs_aware_validation_shifted5
```

---

## 2. Convergence scan

The convergence scan calls the single-point validation over a parameter grid and compares norm-level quantities across adjacent `nk` values and shift modes.

Supported shift modes:

```text
noshift  -> [0.0]
shifted2 -> [0.0, 0.5]
shifted5 -> [0.0, 0.2, 0.4, 0.6, 0.8]
```

The scan reports rows with:

```text
rhs_aware_ward_closed
max S-channel residual / RHS
max Schur-effective residual / reference
max eta projection / R_S
legacy zero-RHS residual / K_eff
K_eff norm
R_eff norm / K_eff norm
K_etaeta condition number
```

It also reports adjacent-`nk` and shift-mode relative changes for:

```text
K_eff norm
R_eff norm
eta projection / R_S
K_etaeta condition number
```

Example light scan:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_rhs_aware_convergence_scan.py \
  --model symmetry_bdg_2band \
  --pairings spm dwave \
  --matsubara-indices 1 2 3 \
  --temperature-K 10.0 \
  --q-values 0.01 0.02 0.04 \
  --nk-values 9 13 \
  --shift-modes noshift \
  --output-dir sandbox/finite_q_tmte/outputs/rhs_aware_convergence_light_noshift
```

Example shift comparison at one representative point:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_rhs_aware_convergence_scan.py \
  --model symmetry_bdg_2band \
  --pairings spm dwave \
  --matsubara-indices 1 \
  --temperature-K 10.0 \
  --q-values 0.02 \
  --nk-values 13 \
  --shift-modes noshift shifted2 shifted5 \
  --output-dir sandbox/finite_q_tmte/outputs/rhs_aware_convergence_shift_compare_n1_q002_nk13
```

---

## 3. Interpretation

`rhs_aware_ward_closed = True` means the Ward identity closes against the finite-q RHS.  It does not mean the physical response is converged and does not mean Casimir input is valid.

Large values of

```text
legacy zero-RHS residual / K_eff
```

are expected at finite q and should not be interpreted as Ward failure.

Large changes across `nk` or shift modes should be interpreted as physical integral convergence warnings, not Ward-identity failures, provided the RHS-aware residual itself remains small.

---

## 4. Boundary with main validation

This sandbox validator is intentionally not wired into the main validation flow.  It is a staging interface for a later main-flow replacement once the sandbox finite-q calculation path is complete.

Do not change `valid_for_casimir_input` based only on this diagnostic.  A production decision still needs a physical convergence policy and an error budget for the actual Casimir-consumed response combination.
