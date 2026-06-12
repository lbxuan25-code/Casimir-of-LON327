# Stage 4.20 User-run targeted Ward refinement scan

## Purpose

Targeted user-run refinement scan for the Stage 4.19 worst-case Ward residual cluster.

## Summary

| quantity | value |
| --- | --- |
| num_total_cases | 48 |
| num_completed_cases | 48 |
| num_closed | 48 |
| num_acceptable_but_monitor | 0 |
| num_not_closed | 0 |
| max_corrected_norm_global | 2.907889e-07 |
| median_corrected_norm | 1.771098e-09 |
| p95_corrected_norm | 2.907800e-07 |

## Worst cases

| q_case | q_scale | level | order | window | max_norm | status |
| --- | --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1.000000e+00 | 4 | 3 | 5.000000e-02 | 2.907889e-07 | CLOSED |
| q_diag_neg | 1.000000e+00 | 4 | 3 | 5.000000e-02 | 2.907889e-07 | CLOSED |
| q_diag_pos | 1.000000e+00 | 4 | 3 | 1.200000e-01 | 2.907800e-07 | CLOSED |
| q_diag_neg | 1.000000e+00 | 4 | 3 | 1.200000e-01 | 2.907800e-07 | CLOSED |
| q_diag_pos | 1.000000e+00 | 4 | 3 | 8.000000e-02 | 2.907739e-07 | CLOSED |
| q_diag_neg | 1.000000e+00 | 4 | 3 | 8.000000e-02 | 2.907739e-07 | CLOSED |
| q_diag_neg | 5.000000e-01 | 4 | 3 | 8.000000e-02 | 2.884724e-07 | CLOSED |
| q_diag_pos | 5.000000e-01 | 4 | 3 | 8.000000e-02 | 2.884724e-07 | CLOSED |
| q_diag_neg | 5.000000e-01 | 4 | 3 | 1.200000e-01 | 2.884644e-07 | CLOSED |
| q_diag_pos | 5.000000e-01 | 4 | 3 | 1.200000e-01 | 2.884644e-07 | CLOSED |

## Filtering / resume behavior

| quantity | value |
| --- | --- |
| active_case_count | 48 |
| loaded_existing_case_count | 60 |
| loaded_existing_active_case_count | 48 |
| ignored_existing_case_count | 12 |
| newly_computed_case_count | 0 |
| results_used_for_summary_count | 48 |

Summary only includes active CLI grid cases. Old completed cases outside the active grid are ignored, preventing old 0.03 eV scans from polluting no-0.03 summaries.

## Diagnostic decision

| quantity | status |
| --- | --- |
| targeted_refinement_status | TARGETED_REFINEMENT_PASSED |
| dominant_failure_channel | left_spatial_source |
| likely_issue | TARGETED_REFINEMENT_CLOSED |
| recommended_next_action | Proceed to the next validation stage; do not claim Casimir readiness. |
