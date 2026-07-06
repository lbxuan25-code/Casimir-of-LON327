# finite-q Ward validation report

## 当前状态
- diagnostic_run_completed: True
- ward_identity_closed: False
- valid_for_casimir_input: False

## q=0 前置结论
- spm: convention_aware_pass
- dwave: convention_aware_pass

## spm 结论
- q0_precondition_status: convention_aware_pass
- max_closure_residual_norm: 0.01374012824796429
- ward_identity_closed: False

## dwave 结论
- q0_precondition_status: convention_aware_pass
- max_closure_residual_norm: 0.013904430677376702
- ward_identity_closed: False

## Casimir gating
- valid_for_casimir_input: False
- This report is diagnostic-only and does not promote finite-q response data to Casimir input.

## Ward triage
- normal finite-q triage conclusion: normal_contact_or_vertex
- operator identity conclusion: response_assembly_or_collective
- contact cancellation conclusion: spm: direct_dominant; dwave: direct_dominant
- suspected primary layer: normal_contact_or_vertex
- recommended next fix: Inspect normal finite-q contact/vertex assembly before collective-sector changes.
- valid_for_casimir_input: False

## 下一步建议
- Investigate the largest finite-q residual rows before changing any Casimir input gate.

## 主要观察
- The finite-q diagnostic completed, but ward_identity_closed is False.
