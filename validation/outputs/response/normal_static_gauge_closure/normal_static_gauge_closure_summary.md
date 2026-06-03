# Normal-State Static Gauge Closure Diagnostic

This is a normal-state static gauge closure diagnostic. It uses a
Peierls-twist finite-difference free-energy stiffness as an independent
baseline for checking normal kernel conventions.
The Peierls baseline is a stiffness reference; this diagnostic does not
assume clean normal-state stiffness must vanish at finite mesh.

The purpose is not to choose a final response formula. It is to locate
whether static closure failure is tied to normal K_para sign, K_dia
sign/contact convention, the mass operator, or the intra/inter balance.

This diagnostic does not modify the formal response formula.
It does not modify BdG, Casimir, reflection, or finite-q code.
It is not a final optical conductivity or Casimir input.

run_command = `python validation/scripts/numerical_stability/diagnose_normal_static_gauge_closure.py --quick`
quick_test_only=True
benchmark_only=True
local_response=True
normal_static_gauge_closure_diagnostic=True
peierls_twist_diagnostic=True
not_final_response_formula=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

## Parameters
- omega_list=0, 0.0001
- nk_list=6, 8
- temperature_K=30
- eta_eV=0.0001
- twist_list=0.001

## Peierls D_fd Trend At omega=0, twist=0.001
- nk=6: D_fd_xx=0.427185
- nk=8: D_fd_xx=0.416418

## Candidate Convention
dominant_best_candidate=minus_para_plus_dia
minus_para_plus_dia means K_dia - K_para.
largest_mean_static_component_norm=K_dia

## Next Step
Use the Peierls baseline and intra/inter/dia decomposition to decide
which normal-state convention needs analytic review. Do not treat the
best candidate reported here as a formula fix without a derivation.

## Figures
- validation/outputs/response/normal_static_gauge_closure/figures/D_fd_xx_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/candidate_K_xx_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/candidate_error_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/intra_inter_dia_decomposition_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/best_candidate_error_vs_omega.png
