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
| matsubara_indices | [1] |
| q_cases | ['q_diag_pos', 'q_diag_neg'] |
| q_scales | [1.0, 0.5] |
| integration_configs | [{'adaptive_level': 1, 'gauss_order': 2, 'fermi_window_eV': 0.05}] |
| coarse_grid | 8 |
| eta_eV | 1e-10 |
| output_si | False |
| quick | True |
| workers | 2 |
| dry_run | False |
| targeted_configs | True |
| planned_num_cases | 4 |

## 5. Ward and diagonal positivity

| quantity | value |
| --- | --- |
| num_cases | 4 |
| num_convergence_comparisons | 0 |
| max_ward_norm | 0.005107144583804304 |
| min_diag_real | 19.515742620401607 |
| num_comparison_pass | 0 |
| num_comparison_monitor | 0 |
| num_comparison_fail | 0 |

## 6. Offdiag values

| q | scale | n | level | window | rel xy | rel LT | A/S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1.0 | 1 | 1 | 0.05 | 1.871823e-01 | 7.477118e-02 | 1.248090e-16 |
| q_diag_pos | 0.5 | 1 | 1 | 0.05 | 1.945053e-02 | 7.872319e-03 | 3.528637e-16 |
| q_diag_neg | 1.0 | 1 | 1 | 0.05 | 1.871823e-01 | 7.477118e-02 | 4.887741e-17 |
| q_diag_neg | 0.5 | 1 | 1 | 0.05 | 1.945053e-02 | 7.872319e-03 | 1.244252e-16 |

## 7. Convergence comparison against baseline

quick/dry-run 没有 convergence comparison。

## 8. q-sign symmetry stability

| quantity | value |
| --- | --- |
| num_pairs | 2 |
| all_pass | True |
| max_diag_even_error | 5.462374690195077e-16 |
| max_offdiag_odd_error | 5.967991259728111e-15 |

## 9. q-scaling stability

| quantity | value |
| --- | --- |
| num_pairs | 2 |
| all_xy_decrease | True |

## 10. Symmetric vs antisymmetric offdiag

| quantity | value |
| --- | --- |
| max_relative_antisymmetric_to_symmetric | 3.5286367550245017e-16 |
| status | SYMMETRIC_OFFDIAG_DOMINATES |

## 11. Interpretation

若 full targeted run 中 Ward、q-sign、q-scaling 和 convergence 均稳定，则应表述为 stable finite-q lattice tensor effect，而不是 Hall response。即使 L/T 投影不能完全消掉 offdiag，也不自动构成失败；这可能说明 lattice tensor structure 不等价于 simple continuum LT decomposition。

## 12. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_3b_status | STAGE5_3B_FAILED_WARD |
| recommended_next_action | Do not proceed; diagnose Ward closure for targeted offdiag cases. |

## 13. Recommended next step

Do not proceed; diagnose Ward closure for targeted offdiag cases. 本阶段仍未进入 reflection/Casimir，也仍未做 SI scaling。
