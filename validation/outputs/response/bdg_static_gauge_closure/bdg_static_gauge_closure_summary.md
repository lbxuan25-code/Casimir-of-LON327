# BdG Static Gauge-Closure Diagnostic

This is a BdG static gauge-closure diagnostic. It checks whether local
K_para + K_dia cancels in the Delta0 -> 0 BdG normal limit and whether
Delta0 > 0 gives a finite, symmetry-consistent candidate rho_s.

It is not a final optical conductivity, not a final Casimir input, does
not contain finite momentum response, and does not change n0_policy.

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_static_gauge_closure.py --quick --scan-kernel-conventions`
quick_test_only=True
benchmark_only=True
local_response=True
static_gauge_closure_diagnostic=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

## Parameters
- kinds=spm, dwave
- delta0_list=0, 0.04
- omega_list=0, 0.0001
- nk=6
- temperature_K=30
- eta_eV=0.0001

## Delta0=0 Gauge Residual
- dwave, omega=0: gauge_residual=1.60287
- dwave, omega=0.0001: gauge_residual=1.60287
- spm, omega=0: gauge_residual=1.60287
- spm, omega=0.0001: gauge_residual=1.60287

## C4 / Offdiag Diagnostics
- dwave: max_abs_rho_s_anisotropy=2.83743e-16, max_offdiag_ratio=1.22729e-17
- spm: max_abs_rho_s_anisotropy=2.85708e-16, max_offdiag_ratio=1.05449e-17

## Figures
- validation/outputs/response/bdg_static_gauge_closure/figures/gauge_residual_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/rho_s_xx_yy_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/rho_s_anisotropy_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/offdiag_ratio_vs_delta0.png
