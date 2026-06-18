# stageSC_2a_bdg_extended_ward_identity_audit

- status: FAILED
- quick: True
- cases: 24

## Summary

| key | value |
| --- | --- |
| best_C_eta2_by_pairing | `{"dwave": {"imag": 0.08, "real": 0.0}, "onsite_s": {"imag": 0.08, "real": 0.0}, "spm": {"imag": 0.08, "real": 0.0}}` |
| best_extended_residual_by_pairing | `{"dwave": 0.0008575070676606875, "onsite_s": 0.0013935686394770043, "spm": 0.0005057180085970878}` |
| best_phase_vertex_by_pairing | `{"dwave": "midpoint", "onsite_s": "midpoint", "spm": "midpoint"}` |
| best_phase_phase_direct_convention_by_pairing | `{"dwave": "plus", "onsite_s": "plus", "spm": "plus"}` |
| best_schur_selected_ward_by_pairing | `{"dwave": 0.0008687870794557502, "onsite_s": 0.0031165277431388074, "spm": 0.000730342226345355}` |
| onsite_s_passed | False |
| material_pairings_passed | False |

## Failures
- dwave extended Ward residual not clearly below bare Ward
- spm material extended Ward failed
- dwave material extended Ward failed

## Case Diagnostics

### Case 1
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- extended_left_residual_max: 0.00139357
- extended_theta_residual_max: 0.000341565
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.95075e-05
### Case 2
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- extended_left_residual_max: 0.030487
- extended_theta_residual_max: 0.0327199
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.95075e-05
### Case 3
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- extended_left_residual_max: 0.00139357
- extended_theta_residual_max: 0.000341565
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 0.0225288
### Case 4
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- extended_left_residual_max: 0.030487
- extended_theta_residual_max: 0.0327199
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 0.0225288
### Case 5
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- extended_left_residual_max: 0.00139357
- extended_theta_residual_max: 0.000341565
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.95075e-05
### Case 6
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- extended_left_residual_max: 0.030487
- extended_theta_residual_max: 0.0327199
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.95075e-05
### Case 7
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: minus
- extended_left_residual_max: 0.00139357
- extended_theta_residual_max: 0.000341565
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 0.0225288
### Case 8
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: minus
- extended_left_residual_max: 0.030487
- extended_theta_residual_max: 0.0327199
- collective_total_condition_number: 30.9736
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 0.0225288
### Case 9
- pairing: spm
- phase_phase_bubble_abs: 0.00621355
- phase_phase_direct_abs: 0.00624639
- phase_phase_total_abs: 3.28373e-05
- phase_phase_bubble: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.006246391303766731}`
- phase_phase_total: `{"imag": 2.713152409173931e-20, "real": 3.283728258529541e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- extended_left_residual_max: 0.000505718
- extended_theta_residual_max: 6.71136e-05
- collective_total_condition_number: 34.5168
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.28373e-05
### Case 10
- pairing: spm
- phase_phase_bubble_abs: 0.00621355
- phase_phase_direct_abs: 0.00624639
- phase_phase_total_abs: 3.28373e-05
- phase_phase_bubble: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.006246391303766731}`
- phase_phase_total: `{"imag": 2.713152409173931e-20, "real": 3.283728258529541e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- extended_left_residual_max: 0.0261784
- extended_theta_residual_max: 0.0246332
- collective_total_condition_number: 34.5168
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.28373e-05
