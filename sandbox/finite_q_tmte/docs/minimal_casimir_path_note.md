# Minimal sandbox Casimir path

This note documents the first diagnostic-only response-to-Casimir chain after the finite-q Ward handoff.

Status:

```text
single-point diagnostic path only
main validation flow unchanged
main Casimir pipeline unchanged
valid_for_casimir_input: False
```

---

## 1. Purpose

The goal is to verify that the sandbox finite-q Schur-effective response can be passed through the existing physical mapping chain:

```text
sandbox K_eff(q, i xi)
  -> spatial response Pi_ij
  -> sigma_model
  -> sigma_sheet
  -> sigma_tilde
  -> R_TE_TM
  -> single-point Lifshitz trace-log integrand
```

This is not a full Casimir calculation.  It does not perform a q-grid integral, phi integral, Matsubara sum, force calculation, or torque calculation.

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_path.py
```

CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_path.py
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_path.py
```

---

## 3. Scope and restrictions

The v1 path intentionally supports only:

```text
n >= 1
q > 0
q vector = (q, 0)
theta_deg = 0
single separation
single Matsubara point
single q point
```

It intentionally does not support:

```text
n = 0
q = 0
theta != 0
arbitrary q-vector
phi integration
Matsubara summation
Casimir energy/force/torque claim
```

The `theta != 0` and torque path require arbitrary `q_vec = (qx,qy)` support in the sandbox response calculation, because the current sandbox Schur response path is still q-along-x only.

---

## 4. Basis convention in v1

The sandbox response order is

```text
[A0, L, T]
```

The existing reflection helper expects a spatial `xy` conductivity tensor.  In this v1 diagnostic this is allowed only because

```text
q = (q, 0),
L = x,
T = y.
```

Therefore the v1 path treats the sandbox `L/T` spatial block as the lab `x/y` block.  This is not valid for arbitrary q direction or theta-dependent torque geometry.

---

## 5. Existing helpers reused

The path reuses the existing non-sandbox physical mapping helpers:

```text
lno327.electrodynamics.conventions
lno327.electrodynamics.reflection
lno327.casimir.lifshitz_integrand
```

Important conventions:

```text
sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV
sigma_sheet_SI = (e^2 / hbar) * sigma_model
sigma_tilde = sigma_sheet_SI / sigma0
R_TE_TM = [[R_TT, R_TL], [-R_LT, -R_LL]]
trace-log = log det[I - exp(-2 kappa d) R1 @ R2]
```

The current diagnostic uses the same plate on both sides at theta=0:

```text
R1_TE_TM = R2_TE_TM
```

---

## 6. Example commands

Dwave representative point:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_path.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/dwave_n1_q002_nk13_minimal_casimir_shifted5
```

SPM representative point:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_path.py \
  --model symmetry_bdg_2band \
  --pairing spm \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/spm_n1_q002_nk13_minimal_casimir_shifted5
```

---

## 7. Interpretation

A successful run means only:

```text
sandbox K_eff can be converted into a finite TE/TM reflection matrix and a finite single-point trace-log integrand.
```

It does not mean:

```text
Casimir result is converged
n=0 is handled
q=0 is handled
phi/q/n integration is correct
torque geometry is implemented
valid_for_casimir_input can be True
```

---

## 8. Immediate next step

After this single-point path is verified, the next natural extension is arbitrary `q_vec=(qx,qy)` support in the sandbox response layer.  That is needed before theta-dependent plate rotation, phi integration, and torque diagnostics can be meaningful.
