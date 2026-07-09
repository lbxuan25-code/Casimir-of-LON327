# Minimal sandbox Casimir n scan

This note documents the fixed-theta, fixed-separation Matsubara-index scan built on top of the q scan diagnostic.

Status:

```text
n scan diagnostic only
positive Matsubara indices only, n>=1
uses q scan diagnostic
uses q-phi average diagnostic
no n=0 policy
no Matsubara tail extrapolation
not a Matsubara sum
not a full q/phi/n Casimir integral
not a torque calculation
valid_for_casimir_input: False
```

---

## 1. Purpose

The n scan evaluates:

```text
for n in matsubara_indices, n>=1:
  xi_eV = 2*pi*n*k_B*T
  run q scan at that n
  record q-phi diagnostic integral and n-dependence
```

The main row quantity is:

```text
q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic
```

This is a q-trapezoid diagnostic over the q-weighted periodic phi average from q scan.  It omits physical prefactors and is not a production energy.

---

## 2. Files

Pipeline:

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_n_scan.py
```

Standalone CLI:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_n_scan.py
```

Unified CLI subcommand:

```text
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py n-scan
```

Tests:

```text
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_n_scan.py
```

---

## 3. Outputs

The CLI writes:

```text
minimal_casimir_n_scan.json
minimal_casimir_n_scan.csv
```

The row fields include:

```text
matsubara_index
xi_eV
q_trapezoid_integral_of_q_weighted_phi_average_logdet_real_diagnostic
q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic
q_trapezoid_integral_of_q_weighted_phi_integral_logdet_real_diagnostic
q_trapezoid_integral_of_q_weighted_phi_integral_logdet_abs_diagnostic
delta_abs_from_n_min
ratio_abs_to_previous_n
partial_sum_abs_diagnostic_no_prefactor
partial_sum_real_diagnostic_no_prefactor
max_q_weighted_phi_average_logdet_abs_diagnostic
max_range_phi_logdet_abs
max_Rdiff_over_q
all_finite_R1
all_finite_R2
all_finite_logdet
all_kappa_match
```

The summary includes:

```text
min_matsubara_index
max_matsubara_index
min_xi_eV
max_xi_eV
last_to_first_abs_ratio_diagnostic
partial_sum_abs_diagnostic_no_prefactor
partial_sum_real_diagnostic_no_prefactor
max_Rdiff_over_nq
max_range_phi_logdet_abs_over_nq
matsubara_tail_not_estimated
n0_policy_included: False
valid_for_casimir_input: False
```

---

## 4. Example command

Use no-shift by default:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  n-scan \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --matsubara-indices 1 2 3 4 5 \
  --temperature-K 10.0 \
  --q-values 0.00125 0.0025 0.005 0.0075 0.01 0.015 0.02 0.04 0.08 \
  --phi-values 0 30 60 90 120 150 180 210 240 270 300 330 \
  --plate2-theta-deg 45.0 \
  --nk 13 \
  --separation-nm 20.0 \
  --shift-fractions 0.0 \
  --skip-rhs-aware-validation \
  --output-dir sandbox/finite_q_tmte/outputs/diag_n_scan_dwave_n1_5_theta45_phi12_qrefined_noshift
```

This run is heavier than a single q scan because it performs one q scan per Matsubara index.

---

## 5. Interpretation

Use the n scan to check:

```text
whether the positive-n q-phi diagnostic decreases with n
whether the last-to-first ratio is already small
whether max_Rdiff or phi ranges grow at larger n
whether additional positive Matsubara indices are needed
```

Do not interpret the partial sums as a final Casimir energy.  They omit:

```text
Matsubara prefactor
n=0 contribution
large-n tail estimate
production q/phi convergence policy
physical unit normalization
```

---

## 6. Non-goals

This scan does not provide:

```text
n=0 static-limit treatment
Matsubara tail extrapolation
final Matsubara sum
q=0 policy
production Casimir input
torque calculation
```
