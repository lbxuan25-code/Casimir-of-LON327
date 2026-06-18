# stageSC_2a_bdg_extended_ward_identity_audit

- status: FAILED
- quick: True
- cases: 48

## Summary

| key | value |
| --- | --- |
| best_C_theta_by_pairing | `{"dwave": {"imag": 2.0, "real": 0.0}, "onsite_s": {"imag": 2.0, "real": 0.0}, "spm": {"imag": 2.0, "real": 0.0}}` |
| best_extended_residual_by_pairing | `{"dwave": 0.0008575070676606875, "onsite_s": 0.001393568639477003, "spm": 0.000505718008597088}` |
| best_phase_vertex_by_pairing | `{"dwave": "midpoint", "onsite_s": "midpoint", "spm": "midpoint"}` |
| best_phase_phase_direct_convention_by_pairing | `{"dwave": "plus", "onsite_s": "plus", "spm": "plus"}` |
| best_schur_selected_ward_by_pairing | `{"dwave": 0.0009514055314417622, "onsite_s": 0.0031825722872174224, "spm": 0.0007238845635511018}` |
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
- C_theta: `{"imag": 1.0, "real": 0.0}`
- extended_left_residual_max: 0.00757684
- extended_theta_residual_max: 5.31701e-05
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
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
- C_theta: `{"imag": -1.0, "real": -0.0}`
- extended_left_residual_max: 0.0228291
- extended_theta_residual_max: 0.000132185
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 3.95075e-05
### Case 3
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- C_theta: `{"imag": 2.0, "real": 0.0}`
- extended_left_residual_max: 0.00139357
- extended_theta_residual_max: 1.36626e-05
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 3.95075e-05
### Case 4
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- C_theta: `{"imag": -2.0, "real": -0.0}`
- extended_left_residual_max: 0.030487
- extended_theta_residual_max: 0.000171692
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 3.95075e-05
### Case 5
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- C_theta: `{"imag": 1.0, "real": 0.0}`
- extended_left_residual_max: 0.00757684
- extended_theta_residual_max: 0.0226215
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 0.0225288
### Case 6
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- C_theta: `{"imag": -1.0, "real": -0.0}`
- extended_left_residual_max: 0.0228291
- extended_theta_residual_max: 0.0224361
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 0.0225288
### Case 7
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- C_theta: `{"imag": 2.0, "real": 0.0}`
- extended_left_residual_max: 0.00139357
- extended_theta_residual_max: 0.0451503
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 0.0225288
### Case 8
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- C_theta: `{"imag": -2.0, "real": -0.0}`
- extended_left_residual_max: 0.030487
- extended_theta_residual_max: 0.0449649
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 0.0225288
### Case 9
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- C_theta: `{"imag": 1.0, "real": 0.0}`
- extended_left_residual_max: 0.00757684
- extended_theta_residual_max: 5.31701e-05
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 3.95075e-05
### Case 10
- pairing: onsite_s
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- C_theta: `{"imag": -1.0, "real": -0.0}`
- extended_left_residual_max: 0.0228291
- extended_theta_residual_max: 0.000132185
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: applied
- phase_phase_abs: 3.95075e-05
