# Minimal sandbox Casimir q-vector path

This note documents the arbitrary-q-vector extension of the minimal response-to-Casimir chain.

Status:

```text
single-point diagnostic path only
arbitrary nonzero q_vec supported
theta_deg = 0 only
main validation flow unchanged
main Casimir pipeline unchanged
valid_for_casimir_input: False
```

---

## 1. Purpose

The goal is to pass a sandbox Schur-effective response computed at

```text
q_vec = (qx, qy)
```

through the physical mapping chain without relying on the old `q=(q,0)` shortcut:

```text
sandbox K_eff(q_vec, i xi) in local [A0, L, T]
  -> spatial response Pi_LT
  -> sigma_model_LT
  -> sigma_sheet_LT
  -> sigma_tilde_LT
  -> LT tangential-electric reflection
  -> R_TE_TM
  -> single-point Lifshitz trace-log integrand
```

This path is LT-native.  It does not reinterpret the local `L/T` response as a lab `x/y` tensor.

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_qvec_path.py
```

CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_qvec_path.py
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_qvec_path.py
```

---

## 3. Scope and restrictions

Supported:

```text
n >= 1
q_vec != 0
single q_vec point
single Matsubara point
single separation
theta_deg = 0
```

Not supported yet:

```text
n = 0
q = 0
theta != 0
plate rotation / torque geometry
phi integration
q-grid integration
Matsubara summation
Casimir energy/force/torque claim
```

The current RHS-aware Ward validation helper is still scalar `q` / q-along-x only.  Therefore this q-vector path runs the old RHS-aware validation only when `qy=0`; for `qy!=0` it records an explicit diagnostic guard:

```text
existing_rhs_aware_validation_is_scalar_q_along_x_only; not run for qy!=0
```

This is intentional and avoids pretending that the scalar-q Ward guard validates arbitrary q direction.

---

## 4. CLI usage

Polar input:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_qvec_path.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --phi-deg 30.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/dwave_n1_q002_phi30_nk13_minimal_casimir_qvec_shifted5
```

Explicit q-vector input:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_qvec_path.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --qx 0.01732050807568877 \
  --qy 0.01 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/dwave_n1_q002_explicit_qvec_nk13_minimal_casimir_qvec_shifted5
```

---

## 5. Interpretation

A successful run means only:

```text
sandbox K_eff(q_vec) can be converted through LT-native conductivity/reflection helpers into a finite TE/TM reflection matrix and a finite single-point trace-log integrand.
```

It does not mean:

```text
Casimir result is converged
arbitrary-q Ward guard is complete
n=0 is handled
q=0 is handled
phi/q/n integration is correct
torque geometry is implemented
valid_for_casimir_input can be True
```

---

## 6. Next step

The next extension should be a small theta/plate-rotation diagnostic.  That will require evaluating plate 1 and plate 2 at their own crystal-frame `q_vec` values and returning both reflection matrices in a common lab TE/TM basis before taking the trace-log.
