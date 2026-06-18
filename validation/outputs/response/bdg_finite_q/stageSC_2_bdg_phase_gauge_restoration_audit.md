# stageSC_2_bdg_phase_gauge_restoration_audit

- status: FAILED
- quick: True
- cases: 2

## Summary

| key | value |
| --- | --- |
| max_bare_Ward | 0.0131047 |
| max_minus_schur_Ward | 0.0131766 |
| max_plus_schur_Ward | 0.0130328 |
| selected_gauge_restored_Ward | 0.0131766 |
| min_improvement_factor | 0.994544 |
| phase_correction_status | `["applied"]` |
| finite_q_current_vertex_status | `["normal_state_exact_finite_q_peierls_vertex"]` |

## Failures
- spm Ward failed selected Schur-minus criterion
- dwave Ward failed selected Schur-minus criterion

## Monitors
- none

## Case Diagnostics

### Case 1
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
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
### Case 2
- pairing: dwave
- omega_eV: 0.01
- q_model: `[0.01, 0.01]`
- status: FAILED
- max_bare_Ward: 0.000951406
- max_minus_schur_Ward: 0.000951428
- max_plus_schur_Ward: 0.000951383
- selected_gauge_restored_Ward: 0.000951428
- improvement_factor: 0.999976
- phase_phase_abs: 0.00332208
- phase_correction_status: applied
- finite_q_current_vertex_status: normal_state_exact_finite_q_peierls_vertex
