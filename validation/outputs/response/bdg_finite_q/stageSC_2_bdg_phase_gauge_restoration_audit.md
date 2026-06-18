# stageSC_2_bdg_phase_gauge_restoration_audit

- status: FAILED
- quick: True
- cases: 18

## Summary

| key | value |
| --- | --- |
| best_onsite_s_Ward | 0.00318257 |
| best_spm_Ward | 0.000723885 |
| best_dwave_Ward | 0.000786975 |
| best_phase_vertex_by_pairing | `{"dwave": "midpoint", "onsite_s": "midpoint", "spm": "midpoint"}` |
| best_phase_phase_direct_convention_by_pairing | `{"dwave": "plus", "onsite_s": "plus", "spm": "plus"}` |
| best_schur_sign_by_pairing | `{"dwave": "minus", "onsite_s": "minus", "spm": "minus"}` |
| selected_convention | `{"phase_phase_direct_convention": "plus", "phase_phase_direct_included": true, "phase_vertex": "symmetric_kpm", "reason": "default derived convention; pass/fail judged separately", "schur_sign": "minus"}` |
| selected_gauge_restored_Ward | `[0.0031825722872174224, 0.0007238845635511018, 0.0009514055314417622]` |
| finite_q_current_vertex_status | `["normal_state_exact_finite_q_peierls_vertex"]` |

## Failures
- onsite_s toy Ward did not close
- spm material Ward failed
- dwave material Ward failed

## Monitors
- none

## Case Diagnostics

### Case 1
- pairing: onsite_s
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0151817
- max_minus_schur_Ward: 0.0152446
- max_plus_schur_Ward: 0.0151187
- selected_gauge_restored_Ward: 0.0152446
- improvement_factor: 0.995871
- phase_phase_abs: 0.0112446
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0
- phase_phase_total_abs: 0.0112446
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.0}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 2
- pairing: onsite_s
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0151817
- max_minus_schur_Ward: 0.00318257
- max_plus_schur_Ward: 0.0331364
- selected_gauge_restored_Ward: 0.00318257
- improvement_factor: 4.77025
- phase_phase_abs: 3.95075e-05
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 3
- pairing: onsite_s
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0151817
- max_minus_schur_Ward: 0.0152131
- max_plus_schur_Ward: 0.0151503
- selected_gauge_restored_Ward: 0.0152131
- improvement_factor: 0.997935
- phase_phase_abs: 0.0225288
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 4
- pairing: onsite_s
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0151817
- max_minus_schur_Ward: 0.0152446
- max_plus_schur_Ward: 0.0151187
- selected_gauge_restored_Ward: 0.0152446
- improvement_factor: 0.995871
- phase_phase_abs: 0.0112446
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0
- phase_phase_total_abs: 0.0112446
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.0}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 5
- pairing: onsite_s
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0151817
- max_minus_schur_Ward: 0.00318257
- max_plus_schur_Ward: 0.0331364
- selected_gauge_restored_Ward: 0.00318257
- improvement_factor: 4.77025
- phase_phase_abs: 3.95075e-05
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 3.95075e-05
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": 3.950747016275534e-05}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 6
- pairing: onsite_s
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0151817
- max_minus_schur_Ward: 0.0152131
- max_plus_schur_Ward: 0.0151503
- selected_gauge_restored_Ward: 0.0152131
- improvement_factor: 0.997935
- phase_phase_abs: 0.0225288
- phase_phase_bubble_abs: 0.0112446
- phase_phase_direct_abs: 0.0112841
- phase_phase_total_abs: 0.0225288
- phase_phase_bubble: `{"imag": 3.0407159317449316e-21, "real": -0.011244641699151648}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.011284149169314403}`
- phase_phase_total: `{"imag": 3.0407159317449316e-21, "real": -0.02252879086846605}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: minus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 7
- pairing: spm
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0131047
- max_minus_schur_Ward: 0.0131766
- max_plus_schur_Ward: 0.0130328
- selected_gauge_restored_Ward: 0.0131766
- improvement_factor: 0.994544
- phase_phase_abs: 0.00621355
- phase_phase_bubble_abs: 0.00621355
- phase_phase_direct_abs: 0
- phase_phase_total_abs: 0.00621355
- phase_phase_bubble: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.0}`
- phase_phase_total: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 8
- pairing: spm
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0131047
- max_minus_schur_Ward: 0.000723885
- max_plus_schur_Ward: 0.0267129
- selected_gauge_restored_Ward: 0.000723885
- improvement_factor: 18.1033
- phase_phase_abs: 3.28373e-05
- phase_phase_bubble_abs: 0.00621355
- phase_phase_direct_abs: 0.00624639
- phase_phase_total_abs: 3.28373e-05
- phase_phase_bubble: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.006246391303766731}`
- phase_phase_total: `{"imag": 2.713152409173931e-20, "real": 3.283728258529541e-05}`
- phase_vertex: midpoint
- phase_phase_direct_convention: plus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 9
- pairing: spm
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0131047
- max_minus_schur_Ward: 0.0131406
- max_plus_schur_Ward: 0.0130689
- selected_gauge_restored_Ward: 0.0131406
- improvement_factor: 0.997272
- phase_phase_abs: 0.0124599
- phase_phase_bubble_abs: 0.00621355
- phase_phase_direct_abs: 0.00624639
- phase_phase_total_abs: 0.0124599
- phase_phase_bubble: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_phase_direct: `{"imag": 0.0, "real": -0.006246391303766731}`
- phase_phase_total: `{"imag": 2.713152409173931e-20, "real": -0.012459945324948166}`
- phase_vertex: midpoint
- phase_phase_direct_convention: minus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 10
- pairing: spm
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.0131047
- max_minus_schur_Ward: 0.0131766
- max_plus_schur_Ward: 0.0130328
- selected_gauge_restored_Ward: 0.0131766
- improvement_factor: 0.994544
- phase_phase_abs: 0.00621355
- phase_phase_bubble_abs: 0.00621355
- phase_phase_direct_abs: 0
- phase_phase_total_abs: 0.00621355
- phase_phase_bubble: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_phase_direct: `{"imag": 0.0, "real": 0.0}`
- phase_phase_total: `{"imag": 2.713152409173931e-20, "real": -0.0062135540211814356}`
- phase_vertex: symmetric_kpm
- phase_phase_direct_convention: plus
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
