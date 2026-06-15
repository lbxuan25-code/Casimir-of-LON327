# Stage 5.3 双层 sheet 电导 offdiag 对称性 / 收敛性审计

## 1. 边界

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_reflection_casimir: True
- not_casimir_ready_claim: True

## 2. 电导约定

- formula: sigma_model_ij(iOmega) = - response[1:3,1:3] / omega_eV
- normalization: bilayer-normalized 2D sheet conductivity
- si_scaling_applied: False
- bulk_3d_conductivity: False
- single_layer_conductivity: False

## 3. 为什么需要审计 Stage 5.2 offdiag

offdiag 不是自动错误。斜向 finite-q 向量会把 longitudinal/transverse 响应投影到 x/y 坐标中，因此需要单独区分几何混合、Hall-like 反对称响应和数值误差。

## 4. 扫描配置

| quantity | value |
| --- | --- |
| temperature_K | 30.0 |
| matsubara_indices | [1, 2, 4, 8] |
| q_cases | ['qx', 'qy', 'q_diag_pos', 'q_diag_neg'] |
| q_scales | [1.0, 0.5] |
| adaptive_levels | [4] |
| gauss_orders | [5] |
| fermi_windows_eV | [0.05] |
| coarse_grid | 32 |
| eta_eV | 1e-10 |
| output_si | False |
| quick | False |
| workers | 8 |
| dry_run | False |
| planned_num_cases | 32 |

## 5. (x,y) offdiag 汇总

| q | scale | n | rel xy offdiag | rel LT offdiag | A/S |
| --- | --- | --- | --- | --- | --- |
| qx | 1.0 | 1 | 7.815021e-17 | 7.815021e-17 | 7.555479e-02 |
| qx | 0.5 | 1 | 2.966062e-17 | 2.966062e-17 | 1.032363e-01 |
| qy | 1.0 | 1 | 6.276503e-17 | 6.276503e-17 | 1.051880e-01 |
| qy | 0.5 | 1 | 2.512776e-17 | 2.512776e-17 | 5.809780e-01 |
| q_diag_pos | 1.0 | 1 | 1.318659e-01 | 8.842474e-02 | 1.398899e-16 |
| q_diag_pos | 0.5 | 1 | 5.418471e-02 | 3.873275e-02 | 6.146584e-17 |
| q_diag_neg | 1.0 | 1 | 1.318659e-01 | 8.842474e-02 | 1.378348e-16 |
| q_diag_neg | 0.5 | 1 | 5.418471e-02 | 3.873275e-02 | 2.660087e-17 |
| qx | 1.0 | 2 | 5.705367e-17 | 5.705367e-17 | 4.369972e-02 |
| qx | 0.5 | 2 | 3.560181e-17 | 3.560181e-17 | 3.512851e-02 |
| qy | 1.0 | 2 | 8.444128e-18 | 8.444128e-18 | 5.295713e-01 |
| qy | 0.5 | 2 | 3.274462e-17 | 3.274462e-17 | 1.567719e-01 |
| q_diag_pos | 1.0 | 2 | 5.415024e-02 | 3.872619e-02 | 8.208380e-17 |
| q_diag_pos | 0.5 | 2 | 1.681503e-02 | 1.160160e-02 | 2.539579e-16 |
| q_diag_neg | 1.0 | 2 | 5.415024e-02 | 3.872619e-02 | 1.159319e-16 |
| q_diag_neg | 0.5 | 2 | 1.681503e-02 | 1.160160e-02 | 1.666512e-16 |
| qx | 1.0 | 4 | 1.242345e-17 | 1.242345e-17 | 1.602504e-01 |
| qx | 0.5 | 4 | 1.857744e-17 | 1.857744e-17 | 8.258335e-02 |
| qy | 1.0 | 4 | 1.457271e-17 | 1.457271e-17 | 1.083011e+00 |
| qy | 0.5 | 4 | 2.434389e-17 | 2.434389e-17 | 2.600855e-02 |
| q_diag_pos | 1.0 | 4 | 1.678355e-02 | 1.159871e-02 | 8.010069e-17 |
| q_diag_pos | 0.5 | 4 | 4.492622e-03 | 3.039253e-03 | 7.597901e-16 |
| q_diag_neg | 1.0 | 4 | 1.678355e-02 | 1.159871e-02 | 1.024235e-17 |
| q_diag_neg | 0.5 | 4 | 4.492622e-03 | 3.039253e-03 | 3.590447e-16 |
| qx | 1.0 | 8 | 1.547358e-17 | 1.547358e-17 | 9.366669e-02 |
| qx | 0.5 | 8 | 6.653559e-18 | 6.653559e-18 | 1.385093e-01 |
| qy | 1.0 | 8 | 1.577821e-17 | 1.577821e-17 | 1.104559e-01 |
| qy | 0.5 | 8 | 4.044370e-17 | 4.044370e-17 | 6.877503e-02 |
| q_diag_pos | 1.0 | 8 | 4.465156e-03 | 3.038259e-03 | 1.147974e-16 |
| q_diag_pos | 0.5 | 8 | 1.137100e-03 | 7.686089e-04 | 1.192072e-15 |
| q_diag_neg | 1.0 | 8 | 4.465156e-03 | 3.038259e-03 | 3.896983e-16 |
| q_diag_neg | 0.5 | 8 | 1.137100e-03 | 7.686089e-04 | 1.340051e-16 |

## 6. (L/T) 投影汇总

| quantity | value |
| --- | --- |
| max_relative_xy_offdiag_norm | 0.13186592735850317 |
| max_relative_LT_offdiag_norm | 0.0884247391936738 |
| median_LT_to_xy_offdiag_ratio | 0.8575809700520582 |
| lt_projection_reduces_offdiag | False |

## 7. symmetric vs antisymmetric offdiag

$\sigma_{xy}\approx\sigma_{yx}$ 表示 symmetric mixing，区别于 Hall-like antisymmetric response。

## 8. q-sign 对称性

$q_y\to -q_y$ 时 offdiag 变号支持 finite-q geometry interpretation。

| quantity | value |
| --- | --- |
| num_pairs | 8 |
| pairs | [{'matsubara_index': 1, 'q_scale': 1.0, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 3.854012201347457e-15, 'q_sign_offdiag_odd_error': 1.7357817040633285e-15, 'q_sign_symmetry_status': 'PASS'}, {'matsubara_index': 1, 'q_scale': 0.5, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 1.6962899706496778e-15, 'q_sign_offdiag_odd_error': 1.7713519137046962e-15, 'q_sign_symmetry_status': 'PASS'}, {'matsubara_index': 2, 'q_scale': 1.0, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 3.622209287984759e-15, 'q_sign_offdiag_odd_error': 2.052713402141693e-15, 'q_sign_symmetry_status': 'PASS'}, {'matsubara_index': 2, 'q_scale': 0.5, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 1.537246770245228e-15, 'q_sign_offdiag_odd_error': 5.283937785337577e-15, 'q_sign_symmetry_status': 'PASS'}, {'matsubara_index': 4, 'q_scale': 1.0, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 8.216860169117175e-16, 'q_sign_offdiag_odd_error': 2.4984167598864315e-15, 'q_sign_symmetry_status': 'PASS'}, {'matsubara_index': 4, 'q_scale': 0.5, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 6.381826419510314e-16, 'q_sign_offdiag_odd_error': 1.075792140057301e-14, 'q_sign_symmetry_status': 'PASS'}, {'matsubara_index': 8, 'q_scale': 1.0, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 1.4137793115694774e-15, 'q_sign_offdiag_odd_error': 1.0907387847905515e-14, 'q_sign_symmetry_status': 'PASS'}, {'matsubara_index': 8, 'q_scale': 0.5, 'adaptive_level': 4, 'gauss_order': 5, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 7.766928023901513e-16, 'q_sign_offdiag_odd_error': 2.5203131410539558e-14, 'q_sign_symmetry_status': 'PASS'}] |
| max_diag_even_error | 3.854012201347457e-15 |
| max_offdiag_odd_error | 2.5203131410539558e-14 |
| status | PASS |

## 9. 轴向 q 与斜向 q 比较

| quantity | value |
| --- | --- |
| max_axial_relative_offdiag_norm | 7.815021132023996e-17 |
| max_diagonal_relative_offdiag_norm | 0.13186592735850317 |
| axial_smaller_than_diagonal | True |

## 10. q-scaling 趋势

| quantity | value |
| --- | --- |
| num_trends | 8 |

## 11. 收敛趋势

| quantity | value |
| --- | --- |
| num_comparisons | 0 |
| max_relative_difference | None |
| convergence_status | PASS |

## 12. Ward residual 诊断

| q | scale | n | ward max | status |
| --- | --- | --- | --- | --- |
| qx | 1.0 | 1 | 1.018988e-09 | PASS |
| qx | 0.5 | 1 | 1.777819e-11 | PASS |
| qy | 1.0 | 1 | 1.018988e-09 | PASS |
| qy | 0.5 | 1 | 1.777819e-11 | PASS |
| q_diag_pos | 1.0 | 1 | 1.038391e-09 | MONITOR |
| q_diag_pos | 0.5 | 1 | 1.438309e-09 | MONITOR |
| q_diag_neg | 1.0 | 1 | 1.038391e-09 | MONITOR |
| q_diag_neg | 0.5 | 1 | 1.438309e-09 | MONITOR |
| qx | 1.0 | 2 | 1.018988e-09 | PASS |
| qx | 0.5 | 2 | 1.777818e-11 | PASS |
| qy | 1.0 | 2 | 1.018988e-09 | PASS |
| qy | 0.5 | 2 | 1.777818e-11 | PASS |
| q_diag_pos | 1.0 | 2 | 1.038391e-09 | MONITOR |
| q_diag_pos | 0.5 | 2 | 1.438309e-09 | MONITOR |
| q_diag_neg | 1.0 | 2 | 1.038391e-09 | MONITOR |
| q_diag_neg | 0.5 | 2 | 1.438309e-09 | MONITOR |
| qx | 1.0 | 4 | 1.018988e-09 | PASS |
| qx | 0.5 | 4 | 1.777819e-11 | PASS |
| qy | 1.0 | 4 | 1.018988e-09 | PASS |
| qy | 0.5 | 4 | 1.777818e-11 | PASS |
| q_diag_pos | 1.0 | 4 | 1.038391e-09 | MONITOR |
| q_diag_pos | 0.5 | 4 | 1.438309e-09 | MONITOR |
| q_diag_neg | 1.0 | 4 | 1.038391e-09 | MONITOR |
| q_diag_neg | 0.5 | 4 | 1.438309e-09 | MONITOR |
| qx | 1.0 | 8 | 1.018988e-09 | PASS |
| qx | 0.5 | 8 | 1.777818e-11 | PASS |
| qy | 1.0 | 8 | 1.018988e-09 | PASS |
| qy | 0.5 | 8 | 1.777818e-11 | PASS |
| q_diag_pos | 1.0 | 8 | 1.038391e-09 | MONITOR |
| q_diag_pos | 0.5 | 8 | 1.438309e-09 | MONITOR |
| q_diag_neg | 1.0 | 8 | 1.038391e-09 | MONITOR |
| q_diag_neg | 0.5 | 8 | 1.438309e-09 | MONITOR |

## 13. 诊断结论

| quantity | value |
| --- | --- |
| conductivity_symmetry_audit_status | CONDUCTIVITY_SYMMETRY_AUDIT_REQUIRES_FURTHER_SOURCE_SYMMETRY_AUDIT |
| recommended_next_action | Do not proceed; audit source symmetry and finite-q tensor structure. |

## 14. 推荐下一步

Do not proceed; audit source symmetry and finite-q tensor structure. 本审计仍未进入 reflection/Casimir。
