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

## Ward criterion
- criterion_version: contact_aware_v1
- closure_response_name: amplitude_phase_schur
- full_bdg_ward_closed: False
- spm max contact-aware closure residual: 0.015382406517454098
- dwave max contact-aware closure residual: 0.0032596479607990957
- largest blocker: pairing=spm, q=[0.02, 0.0], response=amplitude_phase_schur, residual=0.015382406517454098
- recommended next fix: Inspect BdG collective closure for the largest contact-aware finite-q residual.
- valid_for_casimir_input: False

## Casimir gating
- valid_for_casimir_input: False
- This report is diagnostic-only and does not promote finite-q response data to Casimir input.

## Ward triage
- normal finite-q triage conclusion: normal_contact_or_vertex
- operator identity conclusion: response_assembly_or_collective
- contact cancellation conclusion: spm: direct_dominant; dwave: direct_dominant
- normal contact/direct audit: direct_component_placement_suspicious
- recommended normal contact fix: Audit current-current component placement and density-current index ordering before changing formulas.
- normal Ward convention audit: homogeneous_total_includes_contact_rhs
- recommended Ward convention fix: Separate homogeneous bubble Ward validation from contact-aware physical-kernel validation before changing response formulas.
- normal bubble convergence audit: bubble_residual_numerical_convergence_limited
- recommended normal bubble fix: Introduce shifted/twist mesh averaging or tighter normal bubble quadrature before changing formulas.
- normal bubble audit config: nk=[7, 9, 11], q=[0.005, 0.01, 0.02], omega=[0.005, 0.01, 0.02], mesh_shifts=True
- suspected primary layer: bdg_collective_closure
- recommended next fix: Inspect BdG collective closure for the largest contact-aware finite-q residual.
- valid_for_casimir_input: False

## 下一步建议
- Inspect BdG collective closure for the largest contact-aware finite-q residual.

## 主要观察
- The contact-aware finite-q BdG Ward criterion is not closed for the requested pairings.
