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
| small_delta_trend | `{"dwave": [{"delta0_eV": 0.04, "max_component_difference": 0.005614555323160215}, {"delta0_eV": 0.01, "max_component_difference": 0.0003553224579765249}, {"delta0_eV": 0.003, "max_component_difference": 3.2004208177863484e-05}, {"delta0_eV": 0.001, "max_component_difference": 3.5562698133375026e-06}], "onsite_s": [{"delta0_eV": 0.04, "max_component_difference": 0.1679876046582042}, {"delta0_eV": 0.01, "max_component_difference": 0.012821779897532852}, {"delta0_eV": 0.003, "max_component_difference": 0.0011719592395429827}, {"delta0_eV": 0.001, "max_component_difference": 0.000130400435054001}], "spm": [{"delta0_eV": 0.04, "max_component_difference": 0.030054541244520554}, {"delta0_eV": 0.01, "max_component_difference": 0.002429168528877705}, {"delta0_eV": 0.003, "max_component_difference": 0.00022594752358716735}, {"delta0_eV": 0.001, "max_component_difference": 2.5183220258149826e-05}]}` |
| normal_backend_reference_used_only_for_comparison | True |
| shortcut_reference_diff_not_used_for_pass | 0 |

## Failures
- none

## Case Diagnostics

### Case 1
- pairing: onsite_s
- delta0_eV: 0.04
- max_component_difference: 0.167988
- relative_difference: 2.45581
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.86844e-05
### Case 2
- pairing: onsite_s
- delta0_eV: 0.01
- max_component_difference: 0.0128218
- relative_difference: 0.187442
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 6.21476e-06
### Case 3
- pairing: onsite_s
- delta0_eV: 0.003
- max_component_difference: 0.00117196
- relative_difference: 0.0171329
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 6.12532e-07
### Case 4
- pairing: onsite_s
- delta0_eV: 0.001
- max_component_difference: 0.0001304
- relative_difference: 0.00190633
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
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
- max_component_difference: 0.0300545
- relative_difference: 0.439368
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.27563e-05
### Case 7
- pairing: spm
- delta0_eV: 0.01
- max_component_difference: 0.00242917
- relative_difference: 0.0355121
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 5.81546e-06
### Case 8
- pairing: spm
- delta0_eV: 0.003
- max_component_difference: 0.000225948
- relative_difference: 0.00330313
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 5.76449e-07
### Case 9
- pairing: spm
- delta0_eV: 0.001
- max_component_difference: 2.51832e-05
- relative_difference: 0.000368154
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 6.46163e-08
### Case 10
- pairing: spm
- delta0_eV: 0
- max_component_difference: 8.32668e-17
- relative_difference: 1.21728e-15
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: singular_phase_phase
- phase_phase_abs: 0
