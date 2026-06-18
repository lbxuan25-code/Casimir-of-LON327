# stageSC_3_bdg_normal_limit_audit

- status: PASSED
- quick: True
- cases: 5

## Summary

| key | value |
| --- | --- |
| delta0_eV_list | `[0.04, 0.01, 0.003, 0.001, 0.0]` |
| true_BdG_delta0_0_abs_diff_to_normal | 8.32668e-17 |
| true_BdG_delta0_0_rel_diff_to_normal | 1.21728e-15 |
| small_delta_trend | `[{"delta0_eV": 0.04, "max_component_difference": 1.3140454291280537}, {"delta0_eV": 0.01, "max_component_difference": 0.23565414062611784}, {"delta0_eV": 0.003, "max_component_difference": 0.023388779098701292}, {"delta0_eV": 0.001, "max_component_difference": 0.0026220501322384102}]` |
| normal_backend_reference_used_only_for_comparison | True |
| shortcut_reference_diff_not_used_for_pass | 0 |

## Failures
- none

## Case Diagnostics

### Case 1
- delta0_eV: 0.04
- max_component_difference: 1.31405
- relative_difference: 19.2101
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 0.00621364
### Case 2
- delta0_eV: 0.01
- max_component_difference: 0.235654
- relative_difference: 3.44503
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 0.000485764
### Case 3
- delta0_eV: 0.003
- max_component_difference: 0.0233888
- relative_difference: 0.341921
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 4.46753e-05
### Case 4
- delta0_eV: 0.001
- max_component_difference: 0.00262205
- relative_difference: 0.0383318
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 4.97376e-06
### Case 5
- delta0_eV: 0
- max_component_difference: 8.32668e-17
- relative_difference: 1.21728e-15
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 0
