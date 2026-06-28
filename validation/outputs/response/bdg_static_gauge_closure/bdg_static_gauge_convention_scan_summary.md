# BdG Static Gauge Convention Scan

This is a convention scan diagnostic, not a formal response formula change.
The current BdG response implementation is not changed by this scan.
It recombines already-computed K_para and K_dia with trial prefactors to
compare historical and candidate static stiffness conventions. The formal
BdG total kernel now uses the Peierls/free-energy validated
K_dia - K_para convention.

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

## Stiffness Norm Ratio By Convention
- dwave, current: candidate_stiffness_norm_ratio=1.30144
- spm, current: candidate_stiffness_norm_ratio=1.30144
- dwave, minus_para: candidate_stiffness_norm_ratio=0.698565
- spm, minus_para: candidate_stiffness_norm_ratio=0.698565
- dwave, half_para: candidate_stiffness_norm_ratio=1.15072
- spm, half_para: candidate_stiffness_norm_ratio=1.15072
- dwave, minus_half_para: candidate_stiffness_norm_ratio=0.849282
- spm, minus_half_para: candidate_stiffness_norm_ratio=0.849282
- dwave, minus_dia: candidate_stiffness_norm_ratio=0.698565
- spm, minus_dia: candidate_stiffness_norm_ratio=0.698565
- dwave, minus_para_minus_dia: candidate_stiffness_norm_ratio=1.30144
- spm, minus_para_minus_dia: candidate_stiffness_norm_ratio=1.30144
- dwave, half_both: candidate_stiffness_norm_ratio=0.650718
- spm, half_both: candidate_stiffness_norm_ratio=0.650718
- dwave, minus_half_para_half_dia: candidate_stiffness_norm_ratio=0.349282
- spm, minus_half_para_half_dia: candidate_stiffness_norm_ratio=0.349282

## Best Candidate
best_kind=spm
best_convention=minus_half_para_half_dia
best_para_prefactor=-0.5
best_dia_prefactor=0.5
best_candidate_stiffness_norm_ratio=0.349282
legacy_threshold=0.001
legacy_passes_threshold=False

This scan does not claim static gauge closure is solved or select a
final optical conductivity. The formal response contract is the
separately validated K_dia - K_para convention.
