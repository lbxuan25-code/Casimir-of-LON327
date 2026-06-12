# Stage 4.19 Multi-parameter Ward robustness scan

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no source/observable change
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir
- no Casimir-ready claim

## Corrected Ward residual convention

$$R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu},$$

$$R_R[\mu]=i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.$$

## Scan grid

| quantity | values |
| --- | --- |
| scan_mode | representative_default |
| temperatures_K | [30.0, 100.0, 300.0] |
| matsubara_indices | [1, 2, 4] |
| q_cases | ['qx', 'qy', 'q_diag_pos', 'q_diag_neg'] |
| q_scales | [1.0, 0.5, 0.25, 0.125] |
| adaptive_levels | [3, 4] |
| gauss_orders | [3, 5] |
| fermi_windows_eV | [0.03, 0.05, 0.08] |
| coarse_grid | 32 |

## Summary statistics

| quantity | value |
| --- | --- |
| num_total_cases | 24 |
| num_closed | 2 |
| num_acceptable_but_monitor | 19 |
| num_not_closed | 3 |
| max_corrected_norm_global | 3.018433e-04 |
| median_corrected_norm | 5.983392e-06 |
| p95_corrected_norm | 1.066594e-05 |

## Closure table by temperature and Matsubara index

| temperature_K | matsubara_index | cases | closed | monitor | not_closed | max_norm |
| --- | --- | --- | --- | --- | --- | --- |
| 30.0 | 1 | 20 | 1 | 16 | 3 | 3.018433e-04 |
| 30.0 | 2 | 1 | 0 | 1 | 0 | 9.407399e-06 |
| 30.0 | 4 | 1 | 0 | 1 | 0 | 9.407399e-06 |
| 100.0 | 1 | 1 | 1 | 0 | 0 | 1.493572e-07 |
| 300.0 | 1 | 1 | 0 | 1 | 0 | 4.848133e-06 |

## Closure table by q direction

| q_case | cases | closed | monitor | not_closed | max_norm |
| --- | --- | --- | --- | --- | --- |
| q_diag_neg | 4 | 0 | 3 | 1 | 1.066594e-05 |
| q_diag_pos | 12 | 2 | 8 | 2 | 3.018433e-04 |
| qx | 4 | 0 | 4 | 0 | 7.093548e-06 |
| qy | 4 | 0 | 4 | 0 | 7.093548e-06 |

## Adaptive level / Gauss order / Fermi window comparison

| adaptive_level | gauss_order | fermi_window_eV | cases | closed | monitor | not_closed | max_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 0.03 | 1 | 0 | 0 | 1 | 3.018433e-04 |
| 3 | 3 | 0.05 | 20 | 1 | 17 | 2 | 1.066594e-05 |
| 3 | 3 | 0.08 | 1 | 0 | 1 | 0 | 9.407407e-06 |
| 3 | 5 | 0.05 | 1 | 0 | 1 | 0 | 1.681688e-06 |
| 4 | 3 | 0.05 | 1 | 1 | 0 | 0 | 2.907889e-07 |

## Worst cases

| T | n | q_case | q_scale | level | order | window | max_norm | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30.0 | 1 | q_diag_pos | 1.000000e+00 | 3 | 3 | 3.000000e-02 | 3.018433e-04 | NOT_CLOSED |
| 30.0 | 1 | q_diag_neg | 5.000000e-01 | 3 | 3 | 5.000000e-02 | 1.066594e-05 | NOT_CLOSED |
| 30.0 | 1 | q_diag_pos | 5.000000e-01 | 3 | 3 | 5.000000e-02 | 1.066594e-05 | NOT_CLOSED |
| 30.0 | 1 | q_diag_pos | 1.000000e+00 | 3 | 3 | 8.000000e-02 | 9.407407e-06 | ACCEPTABLE_BUT_MONITOR |
| 30.0 | 1 | q_diag_neg | 1.000000e+00 | 3 | 3 | 5.000000e-02 | 9.407399e-06 | ACCEPTABLE_BUT_MONITOR |
| 30.0 | 1 | q_diag_pos | 1.000000e+00 | 3 | 3 | 5.000000e-02 | 9.407399e-06 | ACCEPTABLE_BUT_MONITOR |
| 30.0 | 2 | q_diag_pos | 1.000000e+00 | 3 | 3 | 5.000000e-02 | 9.407399e-06 | ACCEPTABLE_BUT_MONITOR |
| 30.0 | 4 | q_diag_pos | 1.000000e+00 | 3 | 3 | 5.000000e-02 | 9.407399e-06 | ACCEPTABLE_BUT_MONITOR |
| 30.0 | 1 | qx | 5.000000e-01 | 3 | 3 | 5.000000e-02 | 7.093548e-06 | ACCEPTABLE_BUT_MONITOR |
| 30.0 | 1 | qy | 5.000000e-01 | 3 | 3 | 5.000000e-02 | 7.093548e-06 | ACCEPTABLE_BUT_MONITOR |

## Diagnostic decision

| quantity | status |
| --- | --- |
| robustness_status | ROBUSTNESS_FAILURE |
| closure_threshold | 1.000000e-06 |
| monitor_threshold | 1.000000e-05 |
| dominant_failure_channel | right_spatial_observable |
| likely_issue | ADAPTIVE_QUADRATURE_NEEDS_TARGETED_REFINEMENT |

Stage 4.13 fixed the bubble sign. Stage 4.15 addressed the $C-K$ quadrature issue. Stage 4.17/4.18 fixed the right Ward diagnostic convention. Passing this scan means the normal-state response Ward validation is robust over this diagnostic grid; it is not conductivity, reflection, or Casimir completion.

## Next step

Next: diagnose the worst-case parameter cluster before any downstream conductivity/reflection/Casimir use.

If a future robustness scan passes, the next stage may enter response-to-conductivity validation as an independent check.
