# Minimal sandbox Casimir q scan

This note documents the fixed-theta, fixed-Matsubara q scan diagnostic built on top of the phi scan.

Status:

```text
q scan diagnostic only
single Matsubara index only
single plate angle only
single separation only
uses phi scan diagnostic at each q
q-weighted radial-integrand diagnostic included
not a full q/phi/n Casimir integral
not a torque calculation
valid_for_casimir_input: False
```

---

## 1. Purpose

The q scan evaluates a phi scan at each positive q magnitude:

```text
for q in q_values:
  compute periodic phi average/integral of logdet(q, phi)
  report q * phi_average and q * phi_integral diagnostics
```

The q-weighted columns include the two-dimensional radial measure factor `q`, but no physical prefactor and no Matsubara sum.

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_q_scan.py
```

CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_q_scan.py
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_q_scan.py
```

---

## 3. Outputs

The CLI writes:

```text
minimal_casimir_q_scan.json
minimal_casimir_q_scan.csv
```

The compact row fields include:

```text
q_magnitude
phi_average_logdet_real_diagnostic
phi_average_logdet_abs_diagnostic
phi_integral_logdet_real_diagnostic
phi_integral_logdet_abs_diagnostic
q_weighted_phi_average_logdet_real_diagnostic
q_weighted_phi_average_logdet_abs_diagnostic
q_weighted_phi_integral_logdet_real_diagnostic
q_weighted_phi_integral_logdet_abs_diagnostic
range_phi_logdet_abs
max_Rdiff
max_abs_d_logdet_abs_dphi_rad_diagnostic
all_finite_R1
all_finite_R2
all_finite_logdet
all_kappa_match
d_q_weighted_phi_average_logdet_abs_dq_diagnostic
d_q_weighted_phi_integral_logdet_abs_dq_diagnostic
```

The summary includes trapezoid diagnostics over the supplied q values.  These are not production q integrals.

---

## 4. Shift guidance

The earlier shifted5 setup uses five shifted meshes:

```text
0.0 0.2 0.4 0.6 0.8
```

For a q scan, each q already contains a full phi scan.  Therefore shifted5 can be much slower and should not be the default scouting mode.

Recommended sequence:

```text
1. First scouting scan:
   use shifted2 or a coarse no-shift run to inspect shape and failures.

2. Robustness spot-check:
   rerun only a few representative q values with shifted5.

3. Production-candidate convergence later:
   compare nk, q grid, phi grid, and shift modes systematically.
```

A reasonable first pass is shifted2:

```text
--shift-fractions 0.0 0.5
```

If the run is still too slow, use no-shift:

```text
--shift-fractions 0.0
```

but interpret it only as a structural scan, not as a convergence statement.

---

## 5. Example: fast scouting command

This uses a modest q grid, 12 phi points, and shifted2:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_q_scan.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q-values 0.005 0.01 0.02 0.04 0.08 \
  --phi-values 0 30 60 90 120 150 180 210 240 270 300 330 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.5 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/q_scan_dwave_n1_theta45_nk13_phi12_shifted2
```

---

## 6. Example: shifted5 spot-check

After the fast scan, choose a few representative q values and rerun with shifted5:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_q_scan.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-index 1 \
  --temperature-K 10.0 \
  --q-values 0.01 0.02 0.04 \
  --phi-values 0 30 60 90 120 150 180 210 240 270 300 330 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/q_scan_dwave_n1_theta45_nk13_phi12_shifted5_spot
```

---

## 7. Interpretation

A successful run means only:

```text
the fixed-n, fixed-theta q scan is finite and summarized consistently over the chosen q and phi grids.
```

It does not mean:

```text
q integration is converged
phi integration is converged
Matsubara summation is handled
n=0 or q=0 are handled
torque has been computed
valid_for_casimir_input can be True
```

---

## 8. Next step

Use the fast q scan to identify whether the q-weighted diagnostic decays, grows, or plateaus.  Then compare selected q values across shifted2/shifted5 and nk before attempting any larger q-phi diagnostic integral.
