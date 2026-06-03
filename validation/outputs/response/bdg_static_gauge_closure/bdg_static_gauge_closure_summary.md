# BdG Static Gauge-Closure Diagnostic

This is a BdG static gauge-closure diagnostic. It checks whether local
K_para + K_dia cancels in the Delta0 -> 0 BdG normal limit and whether
Delta0 > 0 gives a finite, symmetry-consistent candidate rho_s.

It is not a final optical conductivity, not a final Casimir input, does
not contain finite momentum response, and does not change n0_policy.

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_static_gauge_closure.py --kinds spm dwave --delta0-list 0.0 1e-05 0.0001 0.001 0.01 0.04 --omega-list 0.0 1e-06 2e-06 5e-06 1e-05 2e-05 5e-05 0.0001 --nk 16 --temperature 30.0 --eta 0.0001 --output-prefix /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/bdg_static_gauge_closure/data/bdg_static_gauge_closure --scan-kernel-conventions`
quick_test_only=False
benchmark_only=True
local_response=True
static_gauge_closure_diagnostic=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

## Parameters
- kinds=spm, dwave
- delta0_list=0, 1e-05, 0.0001, 0.001, 0.01, 0.04
- omega_list=0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001
- nk=16
- temperature_K=30
- eta_eV=0.0001

## Delta0=0 Gauge Residual
- dwave, omega=0: gauge_residual=1.77987
- dwave, omega=1e-06: gauge_residual=1.77987
- dwave, omega=2e-06: gauge_residual=1.77987
- dwave, omega=5e-06: gauge_residual=1.77987
- dwave, omega=1e-05: gauge_residual=1.77987
- dwave, omega=2e-05: gauge_residual=1.77987
- dwave, omega=5e-05: gauge_residual=1.77987
- dwave, omega=0.0001: gauge_residual=1.77987
- spm, omega=0: gauge_residual=1.77987
- spm, omega=1e-06: gauge_residual=1.77987
- spm, omega=2e-06: gauge_residual=1.77987
- spm, omega=5e-06: gauge_residual=1.77987
- spm, omega=1e-05: gauge_residual=1.77987
- spm, omega=2e-05: gauge_residual=1.77987
- spm, omega=5e-05: gauge_residual=1.77987
- spm, omega=0.0001: gauge_residual=1.77987

## C4 / Offdiag Diagnostics
- dwave: max_abs_rho_s_anisotropy=1.63667e-15, max_offdiag_ratio=8.18147e-18
- spm: max_abs_rho_s_anisotropy=2.20482e-15, max_offdiag_ratio=1.86575e-17

## Figures
- validation/outputs/response/bdg_static_gauge_closure/figures/gauge_residual_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/rho_s_xx_yy_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/rho_s_anisotropy_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/offdiag_ratio_vs_delta0.png
