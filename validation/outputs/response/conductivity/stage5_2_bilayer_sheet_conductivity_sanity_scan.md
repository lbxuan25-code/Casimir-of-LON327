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
| quick | True |
| dry_run | False |
| temperature_K | 30.0 |
| matsubara_indices | [1, 2] |
| q_cases | ['q_diag_pos'] |
| adaptive_levels | [1] |
| gauss_orders | [2] |
| fermi_windows_eV | [0.05] |
| coarse_grid | 8 |
| planned_num_cases | 2 |

## Summary statistics

| quantity | value |
| --- | --- |
| num_total_cases | 2 |
| num_pass | 0 |
| num_monitor | 0 |
| num_fail | 2 |
| max_ward_norm | 0.005107144583804304 |
| min_diag_real | 10.615532543767877 |
| max_relative_offdiag_norm | 0.18718232780141028 |
| max_relative_xx_yy_anisotropy | 0.00014594887040660518 |
| worst_offdiag_case | {'q_case': 'q_diag_pos', 'matsubara_index': 1, 'adaptive_level': 1, 'gauss_order': 2, 'fermi_window_eV': 0.05, 'status': 'FAIL'} |
| worst_ward_case | {'q_case': 'q_diag_pos', 'matsubara_index': 1, 'adaptive_level': 1, 'gauss_order': 2, 'fermi_window_eV': 0.05, 'status': 'FAIL'} |
| worst_negative_diag_case | {'q_case': 'q_diag_pos', 'matsubara_index': 2, 'adaptive_level': 1, 'gauss_order': 2, 'fermi_window_eV': 0.05, 'status': 'FAIL'} |

## Conductivity sanity by Matsubara frequency

| q | n | omega_eV | sigma_xx | sigma_yy | status |
| --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 1.624329e-02 | 1.951574e+01+1.252307e-16j | 1.952144e+01-2.273518e-16j | FAIL |
| q_diag_pos | 2 | 3.248658e-02 | 1.061553e+01-5.142663e-17j | 1.061714e+01+1.660081e-16j | FAIL |

## Off-diagonal and anisotropy diagnostics

| q | n | rel offdiag | rel xx/yy anisotropy | freq jump | reasons |
| --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 1.871823e-01 | 1.459489e-04 | 0.000000e+00 | WARD_NOT_CLOSED_FOR_CONDUCTIVITY_POINT |
| q_diag_pos | 2 | 9.628065e-02 | 7.567511e-05 | 4.560529e-01 | WARD_NOT_CLOSED_FOR_CONDUCTIVITY_POINT |

## Ward residual diagnostics

| q | n | left | right | max | points |
| --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 5.107145e-03 | 5.107145e-03 | 5.107145e-03 | 544 |
| q_diag_pos | 2 | 5.107145e-03 | 5.107145e-03 | 5.107145e-03 | 544 |

## Diagnostic decision

| quantity | value |
| --- | --- |
| conductivity_sanity_status | CONDUCTIVITY_SANITY_FAILED |
| recommended_next_action | Do not proceed; diagnose failed conductivity or Ward channel. |

## Recommended next step

Do not proceed; diagnose failed conductivity or Ward channel. finite-q angular dependence should not automatically be treated as an error.
