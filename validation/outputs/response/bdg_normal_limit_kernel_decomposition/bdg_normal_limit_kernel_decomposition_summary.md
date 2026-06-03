# BdG Normal-Limit Kernel Decomposition Diagnostic

This is a BdG normal-limit kernel decomposition diagnostic. It compares
Delta0=0 BdG K_para, K_dia, and K_total against locally constructed
normal-state kernel-level K_para and mass-expectation K_dia on the same
mesh and KuboConfig.

K_total is interpreted as the Peierls/free-energy validated stiffness
kernel K_dia - K_para in the current positive-bubble convention.

The purpose is to locate whether static stiffness mismatch is tied to
the paramagnetic bubble, the diamagnetic/contact term, sign convention,
Nambu redundancy, or occupation convention. It is not a final response
formula selection.

This diagnostic does not modify the formal BdG response formula.
It does not modify Casimir calculations.
It contains no finite momentum response.
It is not a final optical conductivity or Casimir input.

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_normal_limit_kernel_decomposition.py --quick`
quick_test_only=True
benchmark_only=True
local_response=True
normal_limit_decomposition_diagnostic=True
delta0_eV=0.0
not_final_response_formula=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

## Parameters
- kinds=spm, dwave
- omega_list=0, 0.0001
- nk=6
- temperature_K=30
- eta_eV=0.0001

## Delta0=0 Ratios At Lowest Omega (0 eV)

Ratios use K_total = K_dia - K_para.
- dwave: para_ratio_xx=1+0j, dia_ratio_xx=1+0j, total_ratio_xx=1+0j, para_relative_error=6.47653e-16, dia_relative_error=2.56908e-16, total_relative_error=7.40899e-16
- spm: para_ratio_xx=1+0j, dia_ratio_xx=1+0j, total_ratio_xx=1+0j, para_relative_error=6.47653e-16, dia_relative_error=2.56908e-16, total_relative_error=7.40899e-16

## Most Inconsistent Piece
lowest_omega_mean_para_relative_error=6.47653e-16
lowest_omega_mean_dia_relative_error=2.56908e-16
lowest_omega_mean_total_relative_error=7.40899e-16
largest_lowest_omega_relative_error=total_relative_error

## Next Step
Use this decomposition to decide which term needs analytic review.
Do not treat any ratio or sign here as a formula fix without a
separate derivation and validation pass.

## Figures
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/bdg_vs_normal_K_para_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/bdg_vs_normal_K_dia_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/bdg_vs_normal_K_total_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/para_ratio_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/dia_ratio_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/relative_error_vs_omega.png
