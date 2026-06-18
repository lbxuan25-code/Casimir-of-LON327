# stageSC_2b_bdg_amplitude_phase_gauge_restoration_audit

- status: FAILED
- quick: True
- cases: 6

## Summary

| key | value |
| --- | --- |
| onsite_s_passed | False |
| material_pairings_passed | False |
| best_phase_vertex_by_pairing | `{"dwave": "symmetric_kpm", "onsite_s": "midpoint", "spm": "midpoint"}` |
| bare_Ward_by_pairing | `{"dwave": 0.0009514055314417622, "onsite_s": 0.015181670144081064, "spm": 0.01310472217996897}` |
| phase_only_best_Ward_by_pairing | `{"dwave": 0.0007971795209860301, "onsite_s": 0.0031825722872174224, "spm": 0.0007238845635511018}` |
| amplitude_phase_Ward_by_pairing | `{"dwave": 0.0008639003186181315, "onsite_s": 0.0031165277431388074, "spm": 0.000730342226345355}` |
| improvement_by_pairing | `{"dwave": 1.1012908676357493, "onsite_s": 4.8713412474842475, "spm": 17.943262360092778}` |
| condition_number_by_pairing | `{"dwave": 6.317519467814357, "onsite_s": 30.97360585433368, "spm": 34.516787335781046}` |
| best_C_eta2_by_pairing | `{"dwave": {"imag": 0.08, "real": 0.0}, "onsite_s": {"imag": 0.08, "real": 0.0}, "spm": {"imag": 0.08, "real": 0.0}}` |

## Failures
- onsite_s amplitude-phase benchmark failed
- spm material amplitude-phase Ward failed
- dwave material amplitude-phase Ward failed

## Case Diagnostics

### Case 1
- pairing: onsite_s
- status: FAILED
- phase_vertex: midpoint
- amplitude_phase_Ward: 0.00311653
- phase_only_best_Ward: 0.00318257
- bare_Ward: 0.0151817
- collective_total_condition_number: 30.9736
- goldstone_counterterm_Cg: `{"imag": -0.0, "real": 7.052593230821511}`
- extended_Ward_best: 0.00139357
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.95075e-05
### Case 2
- pairing: onsite_s
- status: FAILED
- phase_vertex: symmetric_kpm
- amplitude_phase_Ward: 0.00311653
- phase_only_best_Ward: 0.00318257
- bare_Ward: 0.0151817
- collective_total_condition_number: 30.9736
- goldstone_counterterm_Cg: `{"imag": -0.0, "real": 7.052593230821511}`
- extended_Ward_best: 0.00139357
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.95075e-05
### Case 3
- pairing: spm
- status: FAILED
- phase_vertex: midpoint
- amplitude_phase_Ward: 0.000730342
- phase_only_best_Ward: 0.000723885
- bare_Ward: 0.0131047
- collective_total_condition_number: 34.5168
- goldstone_counterterm_Cg: `{"imag": -0.0, "real": 3.9039945648542136}`
- extended_Ward_best: 0.000505718
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.28373e-05
### Case 4
- pairing: spm
- status: FAILED
- phase_vertex: symmetric_kpm
- amplitude_phase_Ward: 0.000730342
- phase_only_best_Ward: 0.000723885
- bare_Ward: 0.0131047
- collective_total_condition_number: 34.5168
- goldstone_counterterm_Cg: `{"imag": -0.0, "real": 3.9039945648542136}`
- extended_Ward_best: 0.000505718
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.28373e-05
### Case 5
- pairing: dwave
- status: FAILED
- phase_vertex: midpoint
- amplitude_phase_Ward: 0.000868787
- phase_only_best_Ward: 0.000786975
- bare_Ward: 0.000951406
- collective_total_condition_number: 6.35031
- goldstone_counterterm_Cg: `{"imag": -0.0, "real": 2.076494224183955}`
- extended_Ward_best: 0.000857507
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.12139e-07
### Case 6
- pairing: dwave
- status: FAILED
- phase_vertex: symmetric_kpm
- amplitude_phase_Ward: 0.0008639
- phase_only_best_Ward: 0.00079718
- bare_Ward: 0.000951406
- collective_total_condition_number: 6.31752
- goldstone_counterterm_Cg: `{"imag": -0.0, "real": 2.076494224183955}`
- extended_Ward_best: 0.000857508
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
- phase_correction_status: amplitude_phase_applied
- phase_phase_abs: 3.5366e-07
