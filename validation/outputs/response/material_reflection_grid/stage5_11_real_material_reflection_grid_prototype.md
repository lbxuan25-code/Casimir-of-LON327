# Stage 5.11 real-material reflection-grid prototype

## 1. Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- real_material_discrete_points_only: True
- no_full_matsubara_sum: True
- no_full_Q_integral: True
- no_energy_output: True
- no_force_output: True
- no_torque_output: True
- not_production_run: True
- not_casimir_ready_claim: True

## 2. Input source

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/casimir_toy_integration/stage5_10_toy_casimir_integration_convergence_audit.json |
| input_stage | Stage 5.10 |
| input_status | STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED |

## 3. Prototype scope

| quantity | value |
| --- | --- |
| real_material_discrete_points | True |
| full_integration_grid | False |
| no_full_matsubara_sum | True |
| no_full_Q_integral | True |
| no_energy_output | True |
| no_force_output | True |
| no_torque_output | True |
| not_production_run | True |
| dry_run_grid_only | False |

## 4. Prototype grid

| quantity | value |
| --- | --- |
| temperature_K | 10.0 |
| separation_nm | 100.0 |
| n_values | [1, 2] |
| Q_nm_inv_values | [0.05, 0.1] |
| phi_deg_values | [0.0, 90.0] |
| num_requested_points | 8 |
| smoke | True |
| n0_excluded | True |
| Q0_excluded | True |
| zero_mode_note | n=0 excluded in Stage 5.11; zero-mode audit is deferred to a later stage. |

## 5. Lattice convention

| quantity | value |
| --- | --- |
| a_x_m | 3.754e-10 |
| a_y_m | 3.754e-10 |
| source | Default in-plane lattice constant for coherently strained thin-film LNO327 / (La,Pr)327-type films on SrLaAlO4-like substrate. Use as a thin-film working value, not as relaxed bulk La3Ni2O7. |
| is_placeholder | False |

## 6. Response numerical config

| quantity | value |
| --- | --- |
| adaptive_level | 1 |
| gauss_order | 2 |
| fermi_window_eV | 0.05 |
| coarse_grid | 8 |
| eta_eV | 1e-10 |

## 7. Pointwise results summary

| quantity | value |
| --- | --- |
| num_success | 8 |
| num_failed | 0 |
| num_monitor | 0 |
| num_diagnostic_fail | 8 |
| max_ward_residual | 0.0068178309656954225 |
| max_abs_sigma_tilde | 6.432641502520613 |
| max_abs_R_TE_TM | 0.9998287242207828 |
| max_abs_logdet | 4.538548177747407e-05 |
| max_abs_logdet_imag | 4.649299137936956e-26 |

## 8. Ward residual summary

Corrected Ward residuals are recorded per point when real response is run.

## 9. Conductivity summary

`sigma_tilde_xy`, `sigma_tilde_LT`, `R_E_LT`, and `R_TE_TM` are retained per point in JSON.

## 10. Reflection matrix summary

TE/TM ordering is `['s', 'p']`; rows are reflected polarization and columns are incident polarization.

## 11. Integrand hook-in summary

Identical-sheet `logdet` values are pointwise hook-in checks only, not an energy integral.

## 12. q-direction diagnostic spot checks

| quantity | value |
| --- | --- |
| num_diagnostics | 0 |

## 13. What this is not

This is not a production grid, not a full Matsubara sum, not a full Q integral, and not Casimir energy/force/torque.

## 14. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_11_status | STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_FAILED |
| recommended_next_action | Proceed to small real-material energy-integration prototype only after material grid convergence strategy is defined. |

## 15. Recommended next step

Proceed to small real-material energy-integration prototype only after material grid convergence strategy is defined.
