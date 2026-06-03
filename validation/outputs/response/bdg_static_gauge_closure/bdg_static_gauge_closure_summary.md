# BdG Static Stiffness Diagnostic

This is a BdG static stiffness diagnostic. In the current positive-bubble
convention, K_total means K_dia - K_para. Clean normal-state stiffness
can be nonzero, so Delta0=0 is checked against the normal dia-minus-para
reference rather than against zero.

For Delta0 > 0 the same K_total is reported as a candidate static
superconducting stiffness rho_s, with C4/offdiag and Delta0-dependence
diagnostics.

It is not a final optical conductivity, not a final Casimir input, does
not contain finite momentum response, and does not change n0_policy.

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_static_gauge_closure.py --kinds spm dwave --delta0-list 0.0 1e-05 0.0001 0.001 0.01 0.04 --omega-list 0.0 1e-06 2e-06 5e-06 1e-05 2e-05 5e-05 0.0001 --nk 16 --temperature 30.0 --eta 0.0001 --output-prefix validation/outputs/response/bdg_static_gauge_closure/data/bdg_static_gauge_closure`
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

## Delta0=0 Normal-Reference Mismatch
- dwave, omega=0: stiffness_reference_mismatch=5.51756e-15, legacy_stiffness_norm_ratio=0.610063
- dwave, omega=1e-06: stiffness_reference_mismatch=1.75782e-15, legacy_stiffness_norm_ratio=0.610063
- dwave, omega=2e-06: stiffness_reference_mismatch=4.74778e-15, legacy_stiffness_norm_ratio=0.610063
- dwave, omega=5e-06: stiffness_reference_mismatch=5.07772e-15, legacy_stiffness_norm_ratio=0.610063
- dwave, omega=1e-05: stiffness_reference_mismatch=6.56174e-15, legacy_stiffness_norm_ratio=0.610063
- dwave, omega=2e-05: stiffness_reference_mismatch=3.60281e-15, legacy_stiffness_norm_ratio=0.610063
- dwave, omega=5e-05: stiffness_reference_mismatch=5.94617e-15, legacy_stiffness_norm_ratio=0.610063
- dwave, omega=0.0001: stiffness_reference_mismatch=4.8819e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=0: stiffness_reference_mismatch=5.51756e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=1e-06: stiffness_reference_mismatch=1.75782e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=2e-06: stiffness_reference_mismatch=4.74778e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=5e-06: stiffness_reference_mismatch=5.07772e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=1e-05: stiffness_reference_mismatch=6.56174e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=2e-05: stiffness_reference_mismatch=3.60281e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=5e-05: stiffness_reference_mismatch=5.94617e-15, legacy_stiffness_norm_ratio=0.610063
- spm, omega=0.0001: stiffness_reference_mismatch=4.8819e-15, legacy_stiffness_norm_ratio=0.610063

## Candidate rho_s C4 / Offdiag Diagnostics
- dwave: max_abs_rho_s_anisotropy=1.7506e-15, max_offdiag_ratio=1.83332e-17
- spm: max_abs_rho_s_anisotropy=2.22126e-15, max_offdiag_ratio=1.82301e-17

## Figures
- validation/outputs/response/bdg_static_gauge_closure/figures/gauge_residual_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/rho_s_xx_yy_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/rho_s_anisotropy_vs_delta0.png
- validation/outputs/response/bdg_static_gauge_closure/figures/offdiag_ratio_vs_delta0.png
