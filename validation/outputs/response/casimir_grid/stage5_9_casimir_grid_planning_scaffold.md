# Stage 5.9 Casimir grid planning scaffold

## 1. Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_heavy_response_run: True
- no_full_matsubara_sum: True
- no_full_Q_integral: True
- no_energy_output: True
- no_force_output: True
- no_torque_output: True
- not_casimir_ready_claim: True

## 2. Input source

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/casimir_integrand/stage5_8_casimir_integrand_prototype.json |
| input_stage | Stage 5.8 |
| input_status | STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED |

## 3. Target energy formula

F/A = k_B*T*sum_n' integral d^2Q/(2*pi)^2 logdet[I-exp(-2*kappa*d) R1 R2]

本阶段只规划变量和测度，不执行正式求和积分。

## 4. Matsubara grid

| quantity | value |
| --- | --- |
| temperature_K | 10.0 |
| n_max | 8 |

Matsubara prime weight 使用 `w0=1/2`，`n>0` 权重为 1。

## 5. (Q, phi) grid

| quantity | value |
| --- | --- |
| Q_m_inv | [0.00000000e+00 7.14285714e+07 1.42857143e+08 2.14285714e+08
 2.85714286e+08 3.57142857e+08 4.28571429e+08 5.00000000e+08] |
| Q_nm_inv | [0.         0.07142857 0.14285714 0.21428571 0.28571429 0.35714286
 0.42857143 0.5       ] |
| phi_rad | [0.         0.52359878 1.04719755 1.57079633 2.0943951  2.61799388
 3.14159265 3.66519143 4.1887902  4.71238898 5.23598776 5.75958653] |
| phi_deg | [  0.  30.  60.  90. 120. 150. 180. 210. 240. 270. 300. 330.] |
| shape_Qx_Qy | [8, 12] |

## 6. Polar measure scaffold

| quantity | value |
| --- | --- |
| min_weight | 0.0 |
| max_weight | 473675425868736.3 |
| sum_weight_scaffold | 2.2736420441699336e+16 |
| quadrature_scope | scaffold only; not production convergence quadrature |

## 7. Round-trip factor summary

| quantity | value |
| --- | --- |
| min | 3.7200401257665066e-44 |
| max | 1.0 |
| has_zero_mode_Q0_factor_one | True |

## 8. Warnings and limitations

- Q=0 has undefined TE/TM in-plane direction and must be handled by symmetry/limit or excluded from angular-grid production runs.
- Existing 8 validation reflection cases are not a production integration grid.
- This scaffold does not perform full Matsubara sum or full Q integration.
- No Casimir energy, force, or torque is output.

## 9. Material response grid requirements

| requirement | value |
| --- | --- |
| required_data | sigma_tilde(i*xi_n,Q,phi) or R_TE_TM(i*xi_n,Q,phi) on a two-dimensional Matsubara/polar grid |
| existing_validation_cases_insufficient | Existing 8 validation reflection cases are not a production integration grid. |
| response_strategy | Use a validated interpolation strategy or compute response directly at every grid point. |
| plate_2_rotation | Q_crystal_2 = R(-theta) Q_lab |
| final_basis | All reflection matrices must be represented in the common lab TE/TM basis. |
| Q_zero_warning | Q=0 has undefined TE/TM in-plane direction and must be handled by symmetry/limit or excluded from angular-grid production runs. |
| cutoff_convergence | High-frequency and large-Q cutoffs require convergence tests. |
| grid_convergence | n_max, Q_max, n_Q, and n_phi require convergence audits. |
| quadrature_warning | Simple scaffold weights are not production quadrature. |

## 10. Checks

| check | status |
| --- | --- |
| matsubara_grid | PASS |
| omega_eV_round_trip | PASS |
| phi_no_duplicate_endpoint | PASS |
| q_grid_shape | PASS |
| polar_measure_nonnegative | PASS |
| round_trip_factor_range | PASS |
| Q0_warning_present | PASS |
| response_grid_insufficiency_warning_present | PASS |

## 11. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_9_status | STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED |
| recommended_next_action | Proceed to toy-model full integration convergence audit before real material production energy. |

## 12. Recommended next step

Proceed to toy-model full integration convergence audit before real material production energy. 当前没有 full Matsubara sum，没有 full Q integral，没有输出 Casimir energy、force 或 torque。
