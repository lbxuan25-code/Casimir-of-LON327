# Minimal sandbox Casimir n-budget diagnostic

This note documents the offline n-budget aggregation diagnostic.

Status:

```text
offline CSV-only diagnostic
does not rerun BdG/q-scan/phi-scan
merges existing n-scan CSV outputs
optionally includes n-tail-fit JSON outputs
known sparse sum is not a dense integer Matsubara sum
log-log interpolation is diagnostic-only
not a Casimir energy
valid_for_casimir_input: False
```

## Purpose

The n-budget diagnostic combines existing positive-n diagnostics into one budget view:

```text
known sparse n terms
missing integer n gaps between known points
log-log interpolated missing-sum diagnostic
optional n-tail-fit estimates
combined diagnostic budget range
```

It is intended to organize current sandbox results, not to define a production Matsubara summation policy.

## Files

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_n_budget.py
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_n_budget.py
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py n-budget
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_n_budget.py
```

## Inputs

Required:

```text
--input-csv one_or_more_minimal_casimir_n_scan.csv
```

Optional:

```text
--tail-fit-json one_or_more_minimal_casimir_n_tail_fit.json
```

Default quantity:

```text
q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic
```

## Outputs

```text
minimal_casimir_n_budget.json
minimal_casimir_n_budget_terms.csv
minimal_casimir_n_budget_gaps.csv
minimal_casimir_n_budget_tail_fits.csv
```

The summary reports:

```text
num_unique_n
min_n
max_n
known_sparse_sum_diagnostic
num_gaps
total_missing_integer_n_between_known_points
has_missing_dense_integer_sum_warning
loglog_interpolated_missing_sum_between_known_points_diagnostic
loglog_interpolated_sum_through_max_known_n_diagnostic
tail_midpoint_min_diagnostic
tail_midpoint_max_diagnostic
loglog_plus_tail_midpoint_min_diagnostic
loglog_plus_tail_midpoint_max_diagnostic
all_finite_logdet_known_terms
all_kappa_match_known_terms
max_Rdiff_over_known_terms
max_range_phi_logdet_abs_over_known_terms
valid_for_casimir_input: False
```

## Example

Combine the existing scans:

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  n-budget \
  --input-csv \
    sandbox/finite_q_tmte/outputs/diag_n_scan_dwave_n1_5_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
    sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n10_100_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
    sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n100_500_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
  --tail-fit-json \
    sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n100_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json \
    sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n150_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json \
    sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n200_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit.json \
  --output-dir sandbox/finite_q_tmte/outputs/diag_n_budget_dwave_theta45_phi12_qrefined_noshift
```

Inspect:

```bash
cat sandbox/finite_q_tmte/outputs/diag_n_budget_dwave_theta45_phi12_qrefined_noshift/minimal_casimir_n_budget_gaps.csv
cat sandbox/finite_q_tmte/outputs/diag_n_budget_dwave_theta45_phi12_qrefined_noshift/minimal_casimir_n_budget_tail_fits.csv
```

## Guardrails

Do not interpret this as:

```text
n=0 policy
validated integer Matsubara dense sum
production tail policy
Casimir energy
torque calculation
```

The output JSON must retain:

```text
valid_for_casimir_input: False
```
