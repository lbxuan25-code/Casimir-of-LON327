# Minimal sandbox Casimir n-tail fit diagnostic

This note documents the offline tail-fit diagnostic for `minimal_casimir_n_scan.csv` outputs.

Status:

```text
offline CSV-only diagnostic
does not rerun BdG/q-scan/phi-scan
fits sparse positive-n diagnostic points
power-law tail fit only
not a Matsubara sum policy
not a production tail policy
not a Casimir energy
valid_for_casimir_input: False
```

## Purpose

The diagnostic reads an existing n-scan CSV and fits a positive quantity, by default:

```text
q_trapezoid_integral_of_q_weighted_phi_average_logdet_abs_diagnostic
```

Supported models:

```text
power_n:   y = A / n^p
power_xi: y = A / xi^p
```

Because `xi = 2*pi*n*k_B*T`, both models should give the same exponent `p`, with different amplitudes.

## Files

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_n_tail_fit.py
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_n_tail_fit.py
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py n-tail-fit
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_n_tail_fit.py
```

## Outputs

```text
minimal_casimir_n_tail_fit.json
minimal_casimir_n_tail_fit_summary.csv
minimal_casimir_n_tail_fit_residuals.csv
```

For a fitted model equivalent to `y_n = A_n / n^p` and `p > 1`, the diagnostic reports integral bounds for:

```text
sum_{n=N+1}^infty A_n/n^p
```

The lower bound uses the integral from `N+1` to infinity; the upper bound uses the integral from `N` to infinity.  The midpoint is their average.  This is diagnostic only, not a validated production Matsubara tail policy.

## Example

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  n-tail-fit \
  --input-csv sandbox/finite_q_tmte/outputs/diag_n_tail_scan_dwave_n100_500_sparse_theta45_phi12_qrefined_noshift/minimal_casimir_n_scan.csv \
  --models power_n power_xi \
  --fit-min-n 100 \
  --tail-start-n-exclusive 500 \
  --output-dir sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n100_500_theta45_phi12_qrefined_noshift
```

Inspect:

```bash
cat sandbox/finite_q_tmte/outputs/diag_n_tail_fit_dwave_n100_500_theta45_phi12_qrefined_noshift/minimal_casimir_n_tail_fit_summary.csv
```

## Guardrails

This does not provide:

```text
n=0 treatment
complete Matsubara sum
production tail policy
Casimir energy
torque calculation
```

The output JSON must retain:

```text
valid_for_casimir_input: False
```
