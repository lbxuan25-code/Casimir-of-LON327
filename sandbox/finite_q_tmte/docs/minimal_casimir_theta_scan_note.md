# Minimal sandbox Casimir theta scan

This note documents the fixed-q, fixed-Matsubara theta scan diagnostic built on top of the single-point theta path.

Status:

```text
theta scan diagnostic only
single lab q_vec only
single Matsubara index only
single separation only
CSV and JSON summary output
finite-difference derivative diagnostic only
not a torque calculation
valid_for_casimir_input: False
```

---

## 1. Purpose

The scan evaluates

```text
log det[I - exp(-2 kappa d) R1(theta1) R2(theta2)]
```

for several plate-2 angles at fixed

```text
q_lab, n, temperature, nk, separation, shift_fractions.
```

It is intended to replace manual shell loops and manual JSON parsing during sandbox diagnostics.

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_theta_scan.py
```

CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_theta_scan.py
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_theta_scan.py
```

---

## 3. Outputs

The CLI writes:

```text
minimal_casimir_theta_scan.json
minimal_casimir_theta_scan.csv
```

The compact row fields include:

```text
theta_deg
theta_rad
relative_theta_deg
logdet_real
logdet_imag
logdet_abs
delta_logdet_real_from_theta0
delta_logdet_abs_from_theta0
d_logdet_real_dtheta_rad_diagnostic
d_logdet_abs_dtheta_rad_diagnostic
Rdiff
R1_norm
R2_norm
p1_Keff_norm
p2_Keff_norm
p1_q_crystal_phi_deg
p2_q_crystal_phi_deg
p1_ward_closed
p2_ward_closed
finite_R1
finite_R2
finite_logdet
kappa_match
```

The derivative columns are finite-difference diagnostics with respect to theta in radians.  They are not physical torque values because the q/phi/n integration and physical prefactors are not included.

---

## 4. Example command

Representative dwave scan:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_theta_scan.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --phi-deg 30.0 \
  --theta-values 0 15 30 45 60 75 90 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/theta_scan_dwave_n1_q002_phi30_nk13_shifted5
```

---

## 5. Interpretation

A successful run means only:

```text
the fixed-q, fixed-n, fixed-distance theta trace-log diagnostic is finite and summarized consistently.
```

It does not mean:

```text
torque has been computed
theta derivative is physically converged
q/phi/n integration is correct
n=0 or q=0 are handled
valid_for_casimir_input can be True
```

---

## 6. Next step

Use the scan to check continuity and symmetry.  If the single-point theta dependence behaves well, the next step is either:

```text
1. add a tiny phi scan at fixed q and n, or
2. build a finite-difference theta-derivative diagnostic over the theta scan output.
```

Both remain diagnostic until q/phi/n integration and convergence are defined.
