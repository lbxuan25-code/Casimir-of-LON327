# Stage 5.12 small real-material energy prototype

## 1. Boundary

- no_response_rerun: True
- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- uses_stage5_11c_reflection_grid: True
- small_grid_only: True
- not_production_energy: True
- no_force_output: True
- no_torque_output: True
- not_casimir_ready_claim: True

## 2. Input source

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/material_reflection_grid/stage5_11c_real_material_reflection_grid_full36_order7_workers8.json |
| input_stage | Stage 5.11 |
| input_status | STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_PASSED |

## 3. Scope and limitations

| quantity | value |
| --- | --- |
| small_grid_only | True |
| not_production_energy | True |
| no_force | True |
| no_torque | True |
| n0_excluded | True |
| zero_mode_not_included | True |
| matsubara_grid_incomplete | True |
| angular_grid_sparse | True |
| Q_grid_sparse | True |

## 4. Grid summary

| quantity | value |
| --- | --- |
| n_values | [1, 2, 4] |
| Q_nm_inv_values | [0.05, 0.1, 0.2] |
| phi_deg_values | [0.0, 45.0, 90.0, 135.0] |
| num_points_used | 36 |

## 5. Energy prototype formula

`F_proto/A = k_B*T*sum_n' sum_Q,phi W_Qphi logdet[I-exp(-2*kappa*d) R R]`。这是稀疏 prototype quadrature。

## 6. Separation scan

| d_nm | F_proto/A | imag | points |
| --- | --- | --- | --- |
| 50.0 | (-1.130697882320405e-09+8.575857801000892e-28j) | 8.575857801000892e-28 | 36 |
| 100.0 | (-7.444254384879716e-12+5.749973186574136e-30j) | 5.749973186574136e-30 | 36 |
| 200.0 | (-3.3791143356842674e-16+2.6103717880570567e-34j) | 2.6103717880570567e-34 | 36 |

## 7. Partial contributions by n

{'1': (-3.7789441760760897e-10+7.33015556556341e-29j), '2': (-3.772589219671436e-10+5.930573741655324e-28j), '4': (-3.755445427456521e-10+1.9122685027892282e-28j)}

## 8. Partial contributions by Q

{'0.05': (-1.1083746190032323e-09+8.591057082475822e-28j), '0.1': (-2.2320564956420332e-11-1.5196192079727825e-30j), '0.2': (-2.6983607519398604e-15-3.089395202186374e-34j)}

## 9. Partial contributions by phi

{'0.0': (-2.82551501123168e-10+4.0193574877993858e-28j), '45.0': (-2.827974400370343e-10+1.0795446234526944e-28j), '90.0': (-2.82551501123168e-10+1.5399383587657042e-28j), '135.0': (-2.827974400370343e-10+1.937017330983108e-28j)}

## 10. Checks

| check | status |
| --- | --- |
| input_status | PASS |
| finite_values | PASS |
| imaginary_part | PASS |
| distance_trend | PASS |
| negative_sign_sanity | PASS |
| warnings_present | PASS |

## 11. What this is not

这不是 production Casimir energy；缺少 n=0，Matsubara/Q/phi 网格极稀疏，因此不是物理预测。不输出 force，也不输出 torque。

## 12. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_12_status | STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_PASSED |
| recommended_next_action | Proceed to material-grid convergence planning; do not interpret prototype energy as physical prediction. |

## 13. Recommended next step

Proceed to material-grid convergence planning; do not interpret prototype energy as physical prediction.
