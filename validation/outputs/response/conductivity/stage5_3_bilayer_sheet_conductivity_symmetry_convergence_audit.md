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
| matsubara_indices | [1] |
| q_cases | ['qx', 'q_diag_pos', 'q_diag_neg'] |
| q_scales | [1.0] |
| adaptive_levels | [1] |
| gauss_orders | [2] |
| fermi_windows_eV | [0.05] |
| coarse_grid | 8 |
| eta_eV | 1e-10 |
| output_si | False |
| quick | True |
| workers | 2 |
| dry_run | False |
| planned_num_cases | 3 |

## 5. (x,y) offdiag 汇总

| q | scale | n | rel xy offdiag | rel LT offdiag | A/S |
| --- | --- | --- | --- | --- | --- |
| qx | 1.0 | 1 | 9.011825e-17 | 9.011825e-17 | 1.442854e-01 |
| q_diag_pos | 1.0 | 1 | 1.871823e-01 | 7.477118e-02 | 1.248090e-16 |
| q_diag_neg | 1.0 | 1 | 1.871823e-01 | 7.477118e-02 | 4.887741e-17 |

## 6. (L/T) 投影汇总

| quantity | value |
| --- | --- |
| max_relative_xy_offdiag_norm | 0.18718232780141064 |
| max_relative_LT_offdiag_norm | 0.07477118420109677 |
| median_LT_to_xy_offdiag_ratio | 0.3994564288164242 |
| lt_projection_reduces_offdiag | False |

## 7. symmetric vs antisymmetric offdiag

$\sigma_{xy}\approx\sigma_{yx}$ 表示 symmetric mixing，区别于 Hall-like antisymmetric response。

## 8. q-sign 对称性

$q_y\to -q_y$ 时 offdiag 变号支持 finite-q geometry interpretation。

| quantity | value |
| --- | --- |
| num_pairs | 1 |
| pairs | [{'matsubara_index': 1, 'q_scale': 1.0, 'adaptive_level': 1, 'gauss_order': 2, 'fermi_window_eV': 0.05, 'q_sign_diag_even_error': 5.462374690195077e-16, 'q_sign_offdiag_odd_error': 1.5833459126281297e-15, 'q_sign_symmetry_status': 'PASS'}] |
| max_diag_even_error | 5.462374690195077e-16 |
| max_offdiag_odd_error | 1.5833459126281297e-15 |
| status | PASS |

## 9. 轴向 q 与斜向 q 比较

| quantity | value |
| --- | --- |
| max_axial_relative_offdiag_norm | 9.011825493739302e-17 |
| max_diagonal_relative_offdiag_norm | 0.18718232780141064 |
| axial_smaller_than_diagonal | True |

## 10. q-scaling 趋势

| quantity | value |
| --- | --- |
| num_trends | 0 |

## 11. 收敛趋势

| quantity | value |
| --- | --- |
| num_comparisons | 0 |
| max_relative_difference | None |
| convergence_status | PASS |

## 12. Ward residual 诊断

| q | scale | n | ward max | status |
| --- | --- | --- | --- | --- |
| qx | 1.0 | 1 | 6.385337e-03 | FAIL |
| q_diag_pos | 1.0 | 1 | 5.107145e-03 | FAIL |
| q_diag_neg | 1.0 | 1 | 5.107145e-03 | FAIL |

## 13. 诊断结论

| quantity | value |
| --- | --- |
| conductivity_symmetry_audit_status | CONDUCTIVITY_SYMMETRY_AUDIT_FAILED_WARD |
| recommended_next_action | Do not proceed; diagnose failed Ward channel. |

## 14. 推荐下一步

Do not proceed; diagnose failed Ward channel. 本审计仍未进入 reflection/Casimir。
