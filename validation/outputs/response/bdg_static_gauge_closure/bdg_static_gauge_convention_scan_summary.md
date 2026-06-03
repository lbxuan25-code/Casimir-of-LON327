# BdG Static Gauge Convention Scan

This is a convention scan diagnostic, not a formal response formula change.
The current BdG response implementation is not changed by this scan.
It recombines already-computed K_para and K_dia with trial prefactors to
locate why the Delta0=0 local static kernel fails to close.

It is not a final response formula, not a final optical conductivity,
not a final Casimir input, and contains no finite momentum response.

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_static_gauge_closure.py --quick --scan-kernel-conventions`
benchmark_only=True
local_response=True
convention_scan_diagnostic=True
not_final_response_formula=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

Delta0=0 lowest_omega=0

## Gauge Residual By Convention
- dwave, current: candidate_gauge_residual=1.60287
- spm, current: candidate_gauge_residual=1.60287
- dwave, minus_para: candidate_gauge_residual=0.39713
- spm, minus_para: candidate_gauge_residual=0.39713
- dwave, half_para: candidate_gauge_residual=1.30144
- spm, half_para: candidate_gauge_residual=1.30144
- dwave, minus_half_para: candidate_gauge_residual=0.698565
- spm, minus_half_para: candidate_gauge_residual=0.698565
- dwave, minus_dia: candidate_gauge_residual=0.39713
- spm, minus_dia: candidate_gauge_residual=0.39713
- dwave, minus_para_minus_dia: candidate_gauge_residual=1.60287
- spm, minus_para_minus_dia: candidate_gauge_residual=1.60287
- dwave, half_both: candidate_gauge_residual=0.801435
- spm, half_both: candidate_gauge_residual=0.801435
- dwave, minus_half_para_half_dia: candidate_gauge_residual=0.198565
- spm, minus_half_para_half_dia: candidate_gauge_residual=0.198565

## Best Candidate
best_kind=spm
best_convention=minus_half_para_half_dia
best_para_prefactor=-0.5
best_dia_prefactor=0.5
best_candidate_gauge_residual=0.198565
acceptable_threshold=0.001
passes_threshold=False

This scan does not claim static gauge closure is solved. A candidate
convention must still be justified analytically and then implemented
as a separate formula change with its own validation.
