# Stage 5.3b finite-q offdiag 收敛性审计

## 1. Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_reflection_casimir: True
- not_casimir_ready_claim: True

## 2. Conductivity convention

- formula: sigma_model_ij(iOmega) = - response[1:3,1:3] / omega_eV
- normalization: bilayer-normalized 2D sheet conductivity
- si_scaling_applied: False
- bulk_3d_conductivity: False
- single_layer_conductivity: False

## 3. 为什么需要 Stage 5.3b

Stage 5.3 已说明 offdiag 更像 finite-q tensor structure，但仍需要 level/window convergence 来排除积分噪声。

## 4. Targeted scan configuration

| quantity | value |
| --- | --- |
| temperature_K | 30.0 |
| matsubara_indices | [1, 2] |
| q_cases | ['q_diag_pos', 'q_diag_neg'] |
| q_scales | [1.0, 0.5] |
| integration_configs | [{'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05}, {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05}, {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08}] |
| coarse_grid | 32 |
| eta_eV | 1e-10 |
| output_si | False |
| quick | False |
| workers | 8 |
| dry_run | False |
| targeted_configs | True |
| planned_num_cases | 24 |

## 5. Ward and diagonal positivity

| quantity | value |
| --- | --- |
| num_cases | 24 |
| num_convergence_comparisons | 16 |
| max_ward_norm | 1.4383158956614426e-09 |
| min_diag_real | 8.394091702176572 |
| num_comparison_pass | 16 |
| num_comparison_monitor | 0 |
| num_comparison_fail | 0 |

## 6. Offdiag values

| q | scale | n | level | window | rel xy | rel LT | A/S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1.0 | 1 | 4 | 0.05 | 1.318659e-01 | 8.842474e-02 | 1.398899e-16 |
| q_diag_pos | 1.0 | 1 | 5 | 0.05 | 1.318659e-01 | 8.842467e-02 | 7.306948e-19 |
| q_diag_pos | 1.0 | 1 | 4 | 0.08 | 1.318659e-01 | 8.842474e-02 | 8.207993e-17 |
| q_diag_pos | 0.5 | 1 | 4 | 0.05 | 5.418471e-02 | 3.873275e-02 | 6.146584e-17 |
| q_diag_pos | 0.5 | 1 | 5 | 0.05 | 5.418470e-02 | 3.873280e-02 | 5.292069e-17 |
| q_diag_pos | 0.5 | 1 | 4 | 0.08 | 5.418471e-02 | 3.873275e-02 | 1.330995e-16 |
| q_diag_neg | 1.0 | 1 | 4 | 0.05 | 1.318659e-01 | 8.842474e-02 | 1.378348e-16 |
| q_diag_neg | 1.0 | 1 | 5 | 0.05 | 1.318659e-01 | 8.842467e-02 | 2.831018e-18 |
| q_diag_neg | 1.0 | 1 | 4 | 0.08 | 1.318659e-01 | 8.842474e-02 | 7.968816e-17 |
| q_diag_neg | 0.5 | 1 | 4 | 0.05 | 5.418471e-02 | 3.873275e-02 | 2.660087e-17 |
| q_diag_neg | 0.5 | 1 | 5 | 0.05 | 5.418470e-02 | 3.873280e-02 | 9.278506e-17 |
| q_diag_neg | 0.5 | 1 | 4 | 0.08 | 5.418471e-02 | 3.873275e-02 | 1.037413e-16 |
| q_diag_pos | 1.0 | 2 | 4 | 0.05 | 5.415024e-02 | 3.872619e-02 | 8.208380e-17 |
| q_diag_pos | 1.0 | 2 | 5 | 0.05 | 5.415024e-02 | 3.872617e-02 | 1.141537e-16 |
| q_diag_pos | 1.0 | 2 | 4 | 0.08 | 5.415024e-02 | 3.872619e-02 | 1.137829e-16 |
| q_diag_pos | 0.5 | 2 | 4 | 0.05 | 1.681503e-02 | 1.160160e-02 | 2.539579e-16 |
| q_diag_pos | 0.5 | 2 | 5 | 0.05 | 1.681503e-02 | 1.160162e-02 | 1.442598e-16 |
| q_diag_pos | 0.5 | 2 | 4 | 0.08 | 1.681503e-02 | 1.160160e-02 | 1.807024e-16 |
| q_diag_neg | 1.0 | 2 | 4 | 0.05 | 5.415024e-02 | 3.872619e-02 | 1.159319e-16 |
| q_diag_neg | 1.0 | 2 | 5 | 0.05 | 5.415024e-02 | 3.872617e-02 | 1.167518e-16 |
| q_diag_neg | 1.0 | 2 | 4 | 0.08 | 5.415024e-02 | 3.872619e-02 | 1.138465e-16 |
| q_diag_neg | 0.5 | 2 | 4 | 0.05 | 1.681503e-02 | 1.160160e-02 | 1.666512e-16 |
| q_diag_neg | 0.5 | 2 | 5 | 0.05 | 1.681503e-02 | 1.160162e-02 | 2.252240e-16 |
| q_diag_neg | 0.5 | 2 | 4 | 0.08 | 1.681503e-02 | 1.160160e-02 | 1.400897e-16 |

## 7. Convergence comparison against baseline

| q | scale | n | comparison | abs d xy | abs d LT | status |
| --- | --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1.0 | 1 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 8.665842e-09 | 6.945376e-08 | PASS |
| q_diag_pos | 1.0 | 1 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 2.019773e-13 | 7.199796e-14 | PASS |
| q_diag_pos | 0.5 | 1 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 3.141333e-09 | 4.957887e-08 | PASS |
| q_diag_pos | 0.5 | 1 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 3.425732e-14 | 6.537826e-14 | PASS |
| q_diag_neg | 1.0 | 1 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 8.665842e-09 | 6.945376e-08 | PASS |
| q_diag_neg | 1.0 | 1 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 2.021716e-13 | 6.820933e-14 | PASS |
| q_diag_neg | 0.5 | 1 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 3.141333e-09 | 4.957887e-08 | PASS |
| q_diag_neg | 0.5 | 1 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 3.413936e-14 | 6.518397e-14 | PASS |
| q_diag_pos | 1.0 | 2 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 7.547388e-10 | 2.080261e-08 | PASS |
| q_diag_pos | 1.0 | 2 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 4.037048e-14 | 3.277933e-14 | PASS |
| q_diag_pos | 0.5 | 2 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 1.414148e-10 | 1.187882e-08 | PASS |
| q_diag_pos | 0.5 | 2 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 2.841477e-15 | 2.156261e-14 | PASS |
| q_diag_neg | 1.0 | 2 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 7.547385e-10 | 2.080261e-08 | PASS |
| q_diag_neg | 1.0 | 2 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 4.055090e-14 | 3.258505e-14 | PASS |
| q_diag_neg | 0.5 | 2 | {'adaptive_level': 5, 'gauss_order': 5, 'fermi_window_eV': 0.05} | 1.414148e-10 | 1.187882e-08 | PASS |
| q_diag_neg | 0.5 | 2 | {'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.08} | 2.758210e-15 | 1.870205e-14 | PASS |

## 8. q-sign symmetry stability

| quantity | value |
| --- | --- |
| num_pairs | 12 |
| all_pass | True |
| max_diag_even_error | 7.706699690207908e-15 |
| max_offdiag_odd_error | 5.283937785337577e-15 |

## 9. q-scaling stability

| quantity | value |
| --- | --- |
| num_pairs | 12 |
| all_xy_decrease | True |

## 10. Symmetric vs antisymmetric offdiag

| quantity | value |
| --- | --- |
| max_relative_antisymmetric_to_symmetric | 2.539579102754088e-16 |
| status | SYMMETRIC_OFFDIAG_DOMINATES |

## 11. Interpretation

若 full targeted run 中 Ward、q-sign、q-scaling 和 convergence 均稳定，则应表述为 stable finite-q lattice tensor effect，而不是 Hall response。即使 L/T 投影不能完全消掉 offdiag，也不自动构成失败；这可能说明 lattice tensor structure 不等价于 simple continuum LT decomposition。

## 12. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_3b_status | STAGE5_3B_PASSED_STABLE_FINITE_Q_LATTICE_TENSOR_EFFECT |
| recommended_next_action | Proceed to Stage 5.4 SI sheet scaling / reflection-input preparation; still do not enter reflection/Casimir. |

## 13. Recommended next step

Proceed to Stage 5.4 SI sheet scaling / reflection-input preparation; still do not enter reflection/Casimir. 本阶段仍未进入 reflection/Casimir，也仍未做 SI scaling。
