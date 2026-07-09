# Minimal sandbox Casimir shift scan

This note documents the fixed-q, fixed-theta, fixed-Matsubara shift scan diagnostic built on top of the phi scan.

Status:

```text
shift scan diagnostic only
single q magnitude only
single plate angle only
single Matsubara index only
single separation only
single-shift phi scans only
R-norm guard included
not a full q/phi/n Casimir integral
not a torque calculation
valid_for_casimir_input: False
```

---

## 1. Purpose

The shift scan evaluates single-shift phi scans:

```text
for shift in shift_values:
  run phi scan with shift_fractions=(shift,)
  report R1_norm, R2_norm, max_R_norm, and large_R_norm per phi
```

It is intended to expose shift-resolved reflection-matrix pathologies that can be hidden by an averaged shifted mesh.

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_shift_scan.py
```

CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_shift_scan.py
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_shift_scan.py
```

---

## 3. Outputs

The CLI writes:

```text
minimal_casimir_shift_scan.json
minimal_casimir_shift_scan.csv
```

The row fields include:

```text
shift_fraction
phi_mod_deg
logdet_abs
delta_logdet_abs_from_shift_phi0
Rdiff
R1_norm
R2_norm
max_R_norm
large_R_norm
p1_Keff_norm
p2_Keff_norm
finite_R1
finite_R2
finite_logdet
kappa_match
```

The summary includes:

```text
num_large_R_norm_rows
has_large_R_norm
max_R_norm
max_Rdiff
range_logdet_abs
worst_R_norm_row
summary_by_shift
```

By default, the warning threshold is:

```text
R_norm_warning_threshold = 2.0
```

This is a diagnostic threshold only.

---

## 4. Example: reproduce q=0.04 shifted-mesh pathology

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_shift_scan.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.04 \
  --phi-values 0 15 30 45 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-values 0.0 0.2 0.4 0.6 0.8 \
  --r-norm-warning-threshold 2.0 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/shift_scan_dwave_n1_q004_theta45_nk13_phi4_shifts002468
```

---

## 5. Suggested interpretation

A large `max_R_norm` means the reflection-matrix diagnostic is shift-sensitive at that q/phi point.  It does not by itself prove a physical instability, and it does not automatically invalidate a small trace-log contribution.

However, if `large_R_norm=True` occurs in the main q-contribution region, then averaged shifted-mesh scans should not be treated as more reliable until mesh/normalization issues are understood.

---

## 6. Current recommended shift policy

For broad diagnostic scans:

```text
prefer no-shift or shifted2
```

For robustness tests:

```text
use shifted5 only as a stress test or spot-check
```

Do not assume shifted5 is more reliable when single-shift `R_norm` values are pathological.

---

## 7. Non-goals

This scan does not handle:

```text
q=0 policy
n=0 policy
q integration convergence
Matsubara summation
torque calculation
production Casimir input
```
