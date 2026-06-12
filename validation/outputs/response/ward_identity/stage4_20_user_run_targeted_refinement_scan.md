# Stage 4.20 User-run targeted Ward refinement scan

## Purpose

Targeted user-run refinement scan for the Stage 4.19 worst-case Ward residual cluster.

## Summary

| quantity | value |
| --- | --- |
| num_total_cases | 24 |
| num_completed_cases | 60 |
| num_closed | 48 |
| num_acceptable_but_monitor | 6 |
| num_not_closed | 6 |
| max_corrected_norm_global | 3.018433e-04 |
| median_corrected_norm | 2.259957e-09 |
| p95_corrected_norm | 5.486262e-05 |

## Worst cases

| q_case | q_scale | level | order | window | max_norm | status |
| --- | --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1.000000e+00 | 3 | 3 | 3.000000e-02 | 3.018433e-04 | NOT_CLOSED |
| q_diag_pos | 1.000000e+00 | 5 | 3 | 3.000000e-02 | 2.986921e-04 | NOT_CLOSED |
| q_diag_pos | 1.000000e+00 | 4 | 3 | 3.000000e-02 | 2.984749e-04 | NOT_CLOSED |
| q_diag_pos | 1.000000e+00 | 3 | 5 | 3.000000e-02 | 4.204093e-05 | NOT_CLOSED |
| q_diag_pos | 1.000000e+00 | 5 | 5 | 3.000000e-02 | 4.184271e-05 | NOT_CLOSED |
| q_diag_pos | 1.000000e+00 | 4 | 5 | 3.000000e-02 | 4.184193e-05 | NOT_CLOSED |
| q_diag_pos | 1.000000e+00 | 3 | 3 | 8.000000e-02 | 9.407407e-06 | ACCEPTABLE_BUT_MONITOR |
| q_diag_pos | 1.000000e+00 | 3 | 3 | 1.200000e-01 | 9.407404e-06 | ACCEPTABLE_BUT_MONITOR |
| q_diag_pos | 1.000000e+00 | 3 | 3 | 5.000000e-02 | 9.407399e-06 | ACCEPTABLE_BUT_MONITOR |
| q_diag_pos | 1.000000e+00 | 3 | 5 | 5.000000e-02 | 1.681688e-06 | ACCEPTABLE_BUT_MONITOR |

## Diagnostic decision

| quantity | status |
| --- | --- |
| targeted_refinement_status | NEEDS_HIGHER_REFINEMENT_OR_WINDOW |
| dominant_failure_channel | right_spatial_observable |
| likely_issue | TARGETED_CLUSTER_STILL_QUADRATURE_LIMITED |
| recommended_next_action | Increase adaptive level, Gauss order, or Fermi window for remaining NOT_CLOSED cases. |
