# BdG Normal-Limit Kernel Decomposition Diagnostic

This is a BdG normal-limit kernel decomposition diagnostic. It compares
Delta0=0 BdG K_para, K_dia, and K_total against locally constructed
normal-state kernel-level K_para and mass-expectation K_dia on the same
mesh and KuboConfig.

The purpose is to locate whether static gauge closure failure is tied to
the paramagnetic bubble, the diamagnetic/contact term, sign convention,
Nambu redundancy, or occupation convention. It is not a final response
formula selection.

This diagnostic does not modify the formal BdG response formula.
It does not modify Casimir calculations.
It contains no finite momentum response.
It is not a final optical conductivity or Casimir input.

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_normal_limit_kernel_decomposition.py --kinds spm dwave --omega-list 0.0 1e-06 2e-06 5e-06 1e-05 2e-05 5e-05 0.0001 --nk 16 --temperature 30.0 --eta 0.0001 --output-prefix /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/bdg_normal_limit_kernel_decomposition/data/bdg_normal_limit_kernel_decomposition`
quick_test_only=False
benchmark_only=True
local_response=True
normal_limit_decomposition_diagnostic=True
delta0_eV=0.0
not_final_response_formula=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

## Parameters
- kinds=spm, dwave
- omega_list=0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001
- nk=16
- temperature_K=30
- eta_eV=0.0001

## Delta0=0 Ratios At Lowest Omega (0 eV)
- dwave: para_ratio_xx=2+0j, dia_ratio_xx=1+0j, total_ratio_xx=1.28054+0j, para_relative_error=1, dia_relative_error=8.79188e-16, total_relative_error=0.280543
- spm: para_ratio_xx=2+0j, dia_ratio_xx=1+0j, total_ratio_xx=1.28054+0j, para_relative_error=1, dia_relative_error=8.79188e-16, total_relative_error=0.280543

## Most Inconsistent Piece
lowest_omega_mean_para_relative_error=1
lowest_omega_mean_dia_relative_error=8.79188e-16
lowest_omega_mean_total_relative_error=0.280543
largest_lowest_omega_relative_error=para_relative_error

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
