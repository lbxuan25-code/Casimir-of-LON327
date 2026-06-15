# Stage 5.2 Bilayer sheet conductivity sanity scan

## Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_reflection_casimir: True
- not_casimir_ready_claim: True

## Conductivity convention

- formula: sigma_model_ij(iOmega) = - response[1:3,1:3] / omega_eV
- normalization: bilayer-normalized 2D sheet conductivity
- si_scaling_applied: False
- bulk_3d_conductivity: False
- single_layer_conductivity: False

This is model-level bilayer sheet conductivity. SI scaling is not applied. This is not reflection/Casimir input yet.

## Scan configuration

| quantity | value |
| --- | --- |
| quick | False |
| dry_run | False |
| temperature_K | 30.0 |
| matsubara_indices | [1, 2, 4, 8] |
| q_cases | ['q_diag_pos', 'q_diag_neg'] |
| adaptive_levels | [4] |
| gauss_orders | [5] |
| fermi_windows_eV | [0.05] |
| coarse_grid | 32 |
| planned_num_cases | 8 |

## Summary statistics

| quantity | value |
| --- | --- |
| num_total_cases | 8 |
| num_pass | 0 |
| num_monitor | 8 |
| num_fail | 0 |
| max_ward_norm | 1.0383913914145193e-09 |
| min_diag_real | 2.8018907302036746 |
| max_relative_offdiag_norm | 0.13186592735850317 |
| max_relative_xx_yy_anisotropy | 0.15772010024885694 |
| worst_offdiag_case | {'q_case': 'q_diag_pos', 'matsubara_index': 1, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'status': 'MONITOR'} |
| worst_ward_case | {'q_case': 'q_diag_pos', 'matsubara_index': 8, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'status': 'MONITOR'} |
| worst_negative_diag_case | {'q_case': 'q_diag_pos', 'matsubara_index': 8, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'status': 'MONITOR'} |

## Conductivity sanity by Matsubara frequency

| q | n | omega_eV | sigma_xx | sigma_yy | status |
| --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 1.624329e-02 | 1.014181e+01+2.780743e-15j | 1.394000e+01-2.542755e-15j | MONITOR |
| q_diag_neg | 1 | 1.624329e-02 | 1.014181e+01+3.511023e-15j | 1.394000e+01-2.263927e-15j | MONITOR |
| q_diag_pos | 2 | 3.248658e-02 | 8.394092e+00-2.561532e-15j | 9.592120e+00-2.101373e-15j | MONITOR |
| q_diag_neg | 2 | 3.248658e-02 | 8.394092e+00+9.811167e-16j | 9.592120e+00-5.857848e-16j | MONITOR |
| q_diag_pos | 4 | 6.497316e-02 | 5.212217e+00+2.446591e-16j | 5.426616e+00-3.083786e-16j | MONITOR |
| q_diag_neg | 4 | 6.497316e-02 | 5.212217e+00+2.978455e-16j | 5.426616e+00+9.278003e-17j | MONITOR |
| q_diag_pos | 8 | 1.299463e-01 | 2.801891e+00+4.584658e-18j | 2.831796e+00-1.034540e-16j | MONITOR |
| q_diag_neg | 8 | 1.299463e-01 | 2.801891e+00-7.269524e-17j | 2.831796e+00+1.286031e-16j | MONITOR |

## Off-diagonal and anisotropy diagnostics

| q | n | rel offdiag | rel xx/yy anisotropy | freq jump | reasons |
| --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 1.318659e-01 | 1.577201e-01 | 0.000000e+00 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT, OFFDIAG_ABOVE_MONITOR_THRESHOLD |
| q_diag_neg | 1 | 1.318659e-01 | 1.577201e-01 | 0.000000e+00 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT, OFFDIAG_ABOVE_MONITOR_THRESHOLD |
| q_diag_pos | 2 | 5.415024e-02 | 6.660815e-02 | 1.723283e-01 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT, OFFDIAG_ABOVE_MONITOR_THRESHOLD |
| q_diag_neg | 2 | 5.415024e-02 | 6.660815e-02 | 1.723283e-01 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT, OFFDIAG_ABOVE_MONITOR_THRESHOLD |
| q_diag_pos | 4 | 1.678355e-02 | 2.015246e-02 | 3.790612e-01 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT, OFFDIAG_ABOVE_MONITOR_THRESHOLD |
| q_diag_neg | 4 | 1.678355e-02 | 2.015246e-02 | 3.790612e-01 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT, OFFDIAG_ABOVE_MONITOR_THRESHOLD |
| q_diag_pos | 8 | 4.465156e-03 | 5.308209e-03 | 4.624378e-01 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT |
| q_diag_neg | 8 | 4.465156e-03 | 5.308209e-03 | 4.624378e-01 | OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT |

## Ward residual diagnostics

| q | n | left | right | max | points |
| --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |
| q_diag_neg | 1 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |
| q_diag_pos | 2 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |
| q_diag_neg | 2 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |
| q_diag_pos | 4 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |
| q_diag_neg | 4 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |
| q_diag_pos | 8 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |
| q_diag_neg | 8 | 1.038391e-09 | 1.038391e-09 | 1.038391e-09 | 1805050 |

## Diagnostic decision

| quantity | value |
| --- | --- |
| conductivity_sanity_status | CONDUCTIVITY_SANITY_MONITOR_OFFDIAG |
| recommended_next_action | Proceed to Stage 5.3 conductivity convergence / symmetry scan before reflection/Casimir. |

## Recommended next step

Proceed to Stage 5.3 conductivity convergence / symmetry scan before reflection/Casimir. finite-q angular dependence should not automatically be treated as an error.
