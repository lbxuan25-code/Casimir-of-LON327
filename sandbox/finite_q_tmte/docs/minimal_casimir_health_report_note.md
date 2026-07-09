# Minimal Casimir health report

This note documents the offline health-report diagnostic.

The health report is intended as a future result-credibility gate for formal data production.  It is **not** a bulk sandbox scan tool.

Status:

```text
offline artifact checker only
reads existing JSON/CSV outputs
does not rerun BdG
does not schedule scans
does not define a production Casimir policy
pass means no configured health threshold was triggered, not physical correctness
valid_for_casimir_input: False
```

---

## 1. Purpose

The tool answers:

```text
Are the existing result artifacts numerically credible enough to inspect or compare?
Which rows/artifacts need local review before being trusted?
Which issues are hard failures versus review warnings?
```

It should eventually sit after formal production runs as a credibility gate:

```text
main runner output -> health report -> pass / needs_review / fail
```

It should not be used to trigger large sandbox scans.

---

## 2. Files

```text
sandbox/finite_q_tmte/tmte/pipeline/minimal_casimir_health_report.py
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_health_report.py
sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py health-report
sandbox/finite_q_tmte/tests/pipeline/test_minimal_casimir_health_report.py
```

---

## 3. Classification levels

The report uses three aggregate statuses:

```text
pass:
  no configured health issue detected

needs_review:
  numerical warning detected, e.g. large R_norm, Rdiff, or phi range
  local convergence/classification is required before interpreting the result

fail:
  hard failure detected, e.g. nonfinite result or kappa mismatch
```

Important guardrail:

```text
health_status = pass does not mean the calculation is physically correct.
It only means this checker did not detect configured numerical-health issues.
```

---

## 4. Default checks

Default thresholds:

```text
R_norm warning threshold: 2.0
Rdiff warning threshold: 2.0
phi-range warning threshold: 1.0e-3
```

The checker looks for columns/keys such as:

```text
all_finite_logdet
all_finite_R1
all_finite_R2
all_kappa_match
kappa_match
R1_norm
R2_norm
max_R_norm
Rdiff
max_Rdiff
max_Rdiff_over_q
max_Rdiff_over_nq
range_phi_logdet_abs
max_range_phi_logdet_abs
max_phi_range
```

Findings include:

```text
nonfinite_result
kappa_mismatch
reflection_norm_pathology
large_Rdiff
large_phi_range
clean
```

---

## 5. Outputs

```text
minimal_casimir_health_report.json
minimal_casimir_health_report_findings.csv
minimal_casimir_health_report.md
```

The JSON summary reports:

```text
health_status
num_input_json
num_input_csv
num_findings
num_fail
num_needs_review
num_pass
max_R_norm_observed
max_Rdiff_observed
max_phi_range_observed
thresholds
valid_for_casimir_input: False
```

---

## 6. Example: check current checkpoint artifacts

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/debug_minimal_casimir_diagnostic.py \
  health-report \
  --input-json \
    sandbox/finite_q_tmte/outputs/diag_n_budget_with_midpoints_dwave_theta45_phi12_qrefined_noshift/minimal_casimir_n_budget.json \
    sandbox/finite_q_tmte/outputs/diag_theta_probe_compare_dwave_theta0_45_90_phi12_qrefined_noshift/theta_probe_compare_summary.json \
  --input-csv \
    sandbox/finite_q_tmte/outputs/diag_phi_scan_dwave_n12_q004_theta45_phi24_noshift/minimal_casimir_phi_scan.csv \
    sandbox/finite_q_tmte/outputs/diag_phi_scan_dwave_n12_q004_theta45_phi24_noshift_nk17/minimal_casimir_phi_scan.csv \
  --output-dir sandbox/finite_q_tmte/outputs/diag_health_report_current_checkpoint
```

Expected interpretation for the current known artifacts:

```text
nk=13 n=12 q=0.04 phi scan:
  needs_review
  reflection_norm_pathology + large_Rdiff + large_phi_range

nk=17 n=12 q=0.04 phi scan:
  pass or much cleaner

n-budget / theta probe:
  may report needs_review if their summaries retain max_Rdiff/phi-range from known local artifacts
```

This is correct: the health report detects artifacts and directs review. It does not apply manual overrides yet.

---

## 7. Future extension: reviewed overrides

A later version should accept a reviewed override manifest, for example:

```text
known_pathologies:
  - id: n12_q004_nk13_grid_artifact
    source_pattern: diag_phi_scan_dwave_n12_q004_theta45_phi24_noshift
    classification: finite_grid_artifact
    resolved_by: nk17 probe
    production_action: do not use nk13 local point without nk convergence
```

That would allow the report to distinguish:

```text
unresolved needs_review
reviewed finite-grid artifact
hard fail
```

The first version intentionally does not implement overrides, so that it remains a simple artifact checker.
