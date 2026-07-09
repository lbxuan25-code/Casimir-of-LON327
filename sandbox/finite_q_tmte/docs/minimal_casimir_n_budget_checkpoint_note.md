# Minimal sandbox Casimir n-budget checkpoint

This note records the current positive-Matsubara diagnostic checkpoint for the finite-q TMTE sandbox path.

Status:

```text
positive-n diagnostic budget checkpoint
n interpolation midpoint validation: globally passed
high-n tail estimate: controlled
local nk=13 pathology: identified and classified as finite-grid artifact
not a Matsubara sum policy
not a Casimir energy
not a torque calculation
valid_for_casimir_input: False
```

---

## 1. Scope

This checkpoint concerns only the positive Matsubara-index diagnostic quantity:

```text
q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic
```

Geometry and scan settings used in the checkpoint:

```text
model: symmetry_bdg_2band
pairing: dwave
temperature_K: 10.0
plate1_theta_deg: 0.0
plate2_theta_deg: 45.0
separation_nm: 20.0
shift_fractions: 0.0
q_values: 0.00125 0.0025 0.005 0.0075 0.01 0.015 0.02 0.04 0.08
phi_values_deg: 0 30 60 90 120 150 180 210 240 270 300 330
nk: 13 for main sparse/midpoint scans
```

This is still a diagnostic budget. It does not include an `n=0` policy, physical Matsubara prefactor, production q/phi convergence, or a production tail policy.

---

## 2. Positive-n scan inputs

The checkpoint combines these n-scan CSVs:

```text
sandbox/finite_q_tmte/outputs/diag_n_scan_dwave_n1_5_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv
sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n10_100_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv
sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n100_500_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv
sandbox/finite_q_tmte/outputs/diag_n_midpoint_scan_dwave_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv
```

Known n values after midpoint validation:

```text
1 2 3 4 5
7 10 12 15 17 20 25 30 40 50 65 80 90 100
125 150 175 200 250 300 400 500
```

The midpoint validation scan added:

```text
7 12 17 25 40 65 90 125 175 250 400
```

These midpoints probe the original sparse gaps:

```text
5-10
10-15
15-20
20-30
30-50
50-80
80-100
100-150
150-200
200-300
300-500
```

---

## 3. Budget before midpoint validation

Using only the original sparse scans and tail fits:

```text
num_unique_n: 16
known_sparse_sum_diagnostic: 0.001054941828913307
loglog_interpolated_missing_sum_between_known_points_diagnostic: 0.006247892292128682
loglog_interpolated_sum_through_max_known_n_diagnostic: 0.007302834121041989
n>500 tail_midpoint_diagnostic range: 4.203618134066705e-05 to 5.9982143385154345e-05
loglog_plus_tail_midpoint_diagnostic range: 0.007344870302382656 to 0.007362816264427144
all_finite_logdet_known_terms: True
all_kappa_match_known_terms: True
valid_for_casimir_input: False
```

---

## 4. Budget after midpoint validation

After adding midpoint n values:

```text
num_unique_n: 27
min_n: 1
max_n: 500
known_sparse_sum_diagnostic: 0.0016111170068615727
num_gaps: 22
total_missing_integer_n_between_known_points: 473
max_missing_integer_n_in_single_gap: 99
loglog_interpolated_missing_sum_between_known_points_diagnostic: 0.005778607425183016
loglog_interpolated_sum_through_max_known_n_diagnostic: 0.007389724432044589
n>500 tail_midpoint_diagnostic range: 4.203618134066705e-05 to 5.9982143385154345e-05
loglog_plus_tail_midpoint_diagnostic range: 0.007431760613385256 to 0.007449706575429743
all_finite_logdet_known_terms: True
all_kappa_match_known_terms: True
valid_for_casimir_input: False
```

The midpoint scan changed the total positive-n diagnostic budget from roughly:

```text
0.00734-0.00736
```

to:

```text
0.00743-0.00745
```

The relative change is about `1.2%`, so the sparse log-log interpolation is globally stable at this diagnostic level.

---

## 5. Tail fits beyond n=500

Tail-fit inputs:

```text
sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n100_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json
sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n150_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json
sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n200_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json
```

Tail-fit summary:

```text
fit_min_n=100:
  best_model: power_n
  p: 2.956140439997479
  r2_log_space: 0.9892156003111957
  rmse_log_space: 0.1715182652372616
  tail_midpoint_estimate_diagnostic: 5.9982143385154345e-05

fit_min_n=150:
  best_model: power_xi
  p: 3.217088923982838
  r2_log_space: 0.9966871513796717
  rmse_log_space: 0.0839477202672282
  tail_midpoint_estimate_diagnostic: 4.713860669247302e-05

fit_min_n=200:
  best_model: power_xi
  p: 3.3763594217105353
  r2_log_space: 0.9987210410475011
  rmse_log_space: 0.045296793916542115
  tail_midpoint_estimate_diagnostic: 4.203618134066705e-05
```

Interpretation:

```text
n=100 still contains some transition-region behavior.
n>=150 and n>=200 fits are cleaner.
The high-n tail is controlled and contributes about 0.6%-0.8% of the current positive-n diagnostic budget.
There is no need to prioritize n=800 or n=1000 before resolving lower-level q/phi/n checkpoint issues.
```

---

## 6. Local nk=13 pathology discovered during midpoint validation

The midpoint n-scan reported:

```text
max_Rdiff_over_nq: 21.397112970075586
max_range_phi_logdet_abs_over_nq: 0.025408277635899174
```

The source was localized to:

```text
n = 12
q = 0.04
nk = 13
```

The q-scan row at `n=12, q=0.04, nk=13` was:

```text
phi_average_logdet_abs_diagnostic: 0.022416518914167734
q_weighted_phi_average_logdet_abs_diagnostic: 0.0008966607565667094
range_phi_logdet_abs: 0.025408277635899174
max_Rdiff: 21.397112970075586
all_finite_logdet: True
all_kappa_match: True
```

The phi-resolved scan showed a clear off-axis reflection-norm pathology.  Off-axis sectors had:

```text
logdet_abs: about 0.0308859447928
Rdiff: about 21.3971129701
R1_norm/R2_norm: about 4.7403315579 and 25.7585770708, swapped by sector
```

Symmetry-axis sectors had:

```text
logdet_abs: about 0.00547766715694
Rdiff: about 0.930584851952
R1_norm/R2_norm: about 0.311971765949 and 1.24255623077, swapped by sector
```

The pathological sectors were the off-axis angles:

```text
15 30 60 75 105 120 150 165 195 210 240 255 285 300 330 345 degrees
```

The clean low sectors were the 45-degree axes:

```text
0 45 90 135 180 225 270 315 degrees
```

Classification at `nk=13`:

```text
local reflection-norm pathology
not a kappa mismatch
not a non-finite logdet failure
not a global n-budget failure
```

---

## 7. nk convergence check resolves the pathology

The same point at `nk=17`:

```text
n = 12
q = 0.04
phi_values: 0 15 ... 345
```

returned:

```text
min_logdet_abs: 0.013989614623704884
max_logdet_abs: 0.013989619067560105
range_logdet_abs: 4.4438552213033056e-09
max_Rdiff: 0.007854728691613981
periodic_phi_average_logdet_abs_diagnostic: 0.01398961758627503
all_finite_logdet: True
all_kappa_match: True
valid_for_casimir_input: False
```

A local q scan at `nk=17` over `q=0.03 0.035 0.04 0.045 0.05` gave:

```text
max_range_phi_logdet_abs: 4.817543611430886e-09
max_Rdiff_over_q: 0.014324522978264179
q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic: 1.250268086893887e-05
all_finite_logdet: True
all_kappa_match: True
```

The corresponding `nk=13` local q scan over the same q window gave:

```text
q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic: 1.3655575692267351e-05
```

The local q-window difference is:

```text
about 1.15e-06
```

Compared with the positive-n budget of about `0.00743-0.00745`, this is about `0.015%`.  Therefore the nk=13 pathology severely contaminates local health metrics, but has negligible impact on the current total positive-n budget.

Final classification:

```text
n=12, q=0.04, nk=13 anomaly:
  finite-grid artifact / finite-mesh reflection pathology
  removed by nk=17
  negligible impact on positive-n diagnostic budget
```

---

## 8. Current conclusion

Current positive-n checkpoint:

```text
n interpolation midpoint validation:
  globally passed
  total diagnostic budget changed by about 1.2%

high-n tail:
  controlled
  n>500 diagnostic tail: about 4.2e-05 to 6.0e-05

local pathology:
  n=12, q=0.04 at nk=13
  classified as finite-grid artifact after nk=17 probe
  affects local Rdiff/phi-range metrics, not total budget materially

positive-n diagnostic budget:
  loglog_plus_tail_midpoint range: 0.007431760613385256 to 0.007449706575429743

valid_for_casimir_input:
  False
```

This checkpoint supports moving from pure positive-n budget validation toward the next diagnostic layer, but not to production Casimir energy.

---

## 9. Recommended next diagnostic

The next diagnostic should be a q/phi/n checkpoint or theta-dependence budget comparison, still diagnostic-only:

```text
keep n=0 excluded until a policy is defined
keep finite-q RHS-aware status explicit
keep no-shift as the primary scan mode
use nk=13 for scouting
apply nk=17 or higher only to suspicious local points
retain valid_for_casimir_input: False
```

Recommended guardrail for future scans:

```text
If Rdiff, R_norm, or phi-range is anomalously large at nk=13,
run a local nk convergence probe before treating the anomaly as physical or as a global blocker.
```
