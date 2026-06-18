# stageSC_3_bdg_normal_limit_audit

- status: PASSED
- quick: True
- cases: 15

## Summary

| key | value |
| --- | --- |
| delta0_eV_list | `[0.04, 0.01, 0.003, 0.001, 0.0]` |
| true_BdG_delta0_0_abs_diff_to_normal | `{"dwave": 8.326680065956345e-17, "onsite_s": 8.326680065956345e-17, "spm": 8.326680065956345e-17}` |
| true_BdG_delta0_0_rel_diff_to_normal | `{"dwave": 1.2172789973372595e-15, "onsite_s": 1.2172789973372595e-15, "spm": 1.2172789973372595e-15}` |
| small_delta_trend | `{"dwave": [{"delta0_eV": 0.04, "max_component_difference": 0.01659500537161005}, {"delta0_eV": 0.01, "max_component_difference": 0.001071107856539444}, {"delta0_eV": 0.003, "max_component_difference": 9.659044195087461e-05}, {"delta0_eV": 0.001, "max_component_difference": 1.0734137884421745e-05}], "onsite_s": [{"delta0_eV": 0.04, "max_component_difference": 0.16605886219429633}, {"delta0_eV": 0.01, "max_component_difference": 0.012806255720933211}, {"delta0_eV": 0.003, "max_component_difference": 0.02450225692855336}, {"delta0_eV": 0.001, "max_component_difference": 0.0027454032424983807}], "spm": [{"delta0_eV": 0.04, "max_component_difference": 0.03001854533278192}, {"delta0_eV": 0.01, "max_component_difference": 0.0024311487336726536}, {"delta0_eV": 0.003, "max_component_difference": 0.02309046376256197}, {"delta0_eV": 0.001, "max_component_difference": 0.00258837974158512}]}` |
| normal_backend_reference_used_only_for_comparison | True |
| shortcut_reference_diff_not_used_for_pass | 0 |

## Failures
- none

## Case Diagnostics

### Case 1
- pairing: onsite_s
- delta0_eV: 0.04
- max_component_difference: 0.166059
- relative_difference: 2.42762
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 3.86844e-05
### Case 2
- pairing: onsite_s
- delta0_eV: 0.01
- max_component_difference: 0.0128063
- relative_difference: 0.187215
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 6.21476e-06
### Case 3
- pairing: onsite_s
- delta0_eV: 0.003
- max_component_difference: 0.0245023
- relative_difference: 0.358199
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 6.12532e-07
### Case 4
- pairing: onsite_s
- delta0_eV: 0.001
- max_component_difference: 0.0027454
- relative_difference: 0.0401351
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 6.86269e-08
### Case 5
- pairing: onsite_s
- delta0_eV: 0
- max_component_difference: 8.32668e-17
- relative_difference: 1.21728e-15
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 0
### Case 6
- pairing: spm
- delta0_eV: 0.04
- max_component_difference: 0.0300185
- relative_difference: 0.438842
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 3.27563e-05
### Case 7
- pairing: spm
- delta0_eV: 0.01
- max_component_difference: 0.00243115
- relative_difference: 0.035541
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 5.81546e-06
### Case 8
- pairing: spm
- delta0_eV: 0.003
- max_component_difference: 0.0230905
- relative_difference: 0.33756
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 5.76449e-07
### Case 9
- pairing: spm
- delta0_eV: 0.001
- max_component_difference: 0.00258838
- relative_difference: 0.0378396
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 6.46163e-08
### Case 10
- pairing: spm
- delta0_eV: 0
- max_component_difference: 8.32668e-17
- relative_difference: 1.21728e-15
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 0
