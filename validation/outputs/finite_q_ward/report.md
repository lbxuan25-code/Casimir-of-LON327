# finite-q Ward validation report

## Current status
- diagnostic_run_completed: True
- ward_identity_closed: False
- valid_for_casimir_input: False

## q=0 preconditions
- spm: convention_aware_pass
- dwave: convention_aware_pass

## spm conclusion
- q0_precondition_status: convention_aware_pass
- max_closure_residual_norm: 0.0017709885262365808
- ward_identity_closed: False

## dwave conclusion
- q0_precondition_status: convention_aware_pass
- max_closure_residual_norm: 0.011058627347831026
- ward_identity_closed: False

## Ward criterion
- criterion_version: full_hessian_v1
- criterion_formal_name: full_hessian_v1
- closure_response_name: amplitude_phase_schur
- full_bdg_ward_closed: False
- largest blocker: pairing=dwave, q=[0.02, 0.0], response=amplitude_phase_schur, primary_residual=0.011058627347831026
- recommended next fix: Inspect the largest homogeneous full-Hessian Schur Ward residual.
- valid_for_casimir_input: False

## Collective block localization
- identity_version: collective_block_v1
- largest block residual: pairing=dwave, q=[0.02, 0.0], block=aa_left, norm=0.012111932873629785
- max_schur_reconstruction_difference_norm: 1.5062217954494666e-18
- diagnostic_role: algebraic localization of the four block identities; not a new Ward criterion

## Casimir gating
- valid_for_casimir_input: False
- This report is diagnostic-only and does not promote finite-q response data to Casimir input.

## Next action
- Inspect the largest homogeneous full-Hessian Schur Ward residual.

## Main observation
- The finite-q BdG full-Hessian Schur Ward criterion is not closed for the requested pairings.
