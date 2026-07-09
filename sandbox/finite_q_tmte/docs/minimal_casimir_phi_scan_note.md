# Minimal sandbox Casimir phi scan

This note documents the fixed-q, fixed-theta, fixed-Matsubara phi scan diagnostic built on top of the single-point theta path.

Status:

```text
phi scan diagnostic only
single q magnitude only
single plate angle only
single Matsubara index only
single separation only
periodic phi-average/integral diagnostic only
not a full q/phi/n Casimir integral
not a torque calculation
valid_for_casimir_input: False
```

---

## 1. Purpose

The scan evaluates the two-plate trace-log diagnostic over lab-frame momentum directions:

```text
q_lab(phi) = q (cos phi, sin phi)
```

at fixed

```text
q magnitude, plate2 theta, n, temperature, nk, separation, shift_fractions.
```

It is intended to inspect angular integrand structure before introducing q-grid and Matsubara integration.

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_phi_scan.py
```

CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_phi_scan.py
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_phi_scan.py
```

---

## 3. Outputs

The CLI writes:

```text
minimal_casimir_phi_scan.json
minimal_casimir_phi_scan.csv
```

The compact row fields include:

```text
phi_deg
phi_mod_deg
phi_rad
logdet_real
logdet_imag
logdet_abs
delta_logdet_real_from_phi0
delta_logdet_abs_from_phi0
d_logdet_real_dphi_rad_diagnostic
d_logdet_abs_dphi_rad_diagnostic
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

The summary includes periodic trapezoid diagnostics:

```text
periodic_phi_integral_logdet_real_diagnostic
periodic_phi_integral_logdet_abs_diagnostic
periodic_phi_average_logdet_real_diagnostic
periodic_phi_average_logdet_abs_diagnostic
```

These are diagnostics only.  They are not the full Casimir integral because q-grid integration, Matsubara summation, n=0, q=0, and physical prefactors are not included.

---

## 4. Periodic convention

The scan treats phi as a periodic variable.  Therefore `0` and `360` degrees are duplicates and must not both appear in one scan.

For multiple phi points, the integral diagnostic uses a periodic wrap-around trapezoid rule.  For one phi point, derivative and integral diagnostics are left unset.

---

## 5. Example command

Representative dwave scan at fixed theta:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_phi_scan.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q 0.02 \
  --phi-values 0 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/phi_scan_dwave_n1_q002_theta45_nk13_shifted5
```

---

## 6. Interpretation

A successful run means only:

```text
the fixed-q, fixed-n, fixed-theta phi trace-log diagnostic is finite and summarized consistently.
```

It does not mean:

```text
full phi integration is converged
q integration is handled
Matsubara summation is handled
n=0 or q=0 are handled
torque has been computed
valid_for_casimir_input can be True
```

---

## 7. Next step

Use the phi scan to check continuity, periodicity, and possible angular cancellations.  If the angular structure is smooth, the next step is a fixed-n q-scan or a q-phi diagnostic integral, still without claiming production readiness.
