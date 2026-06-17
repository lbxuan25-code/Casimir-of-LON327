# Stage 5.10 toy Casimir integration convergence audit

## 1. Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_heavy_response_run: True
- toy_model_only: True
- no_real_material_response_grid: True
- no_real_LNO327_energy: True
- no_force_output: True
- no_torque_output: True
- not_casimir_ready_claim: True

## 2. Input source

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/casimir_grid/stage5_9_casimir_grid_planning_scaffold.json |
| input_stage | Stage 5.9 |
| input_status | STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED |

## 3. Toy model definitions

| model | definition |
| --- | --- |
| zero | {'R': '0'} |
| isotropic_identical | {'r_s': '-r0*f(xi,Q)', 'r_p': '+r0*f(xi,Q)', 'r0': 0.3} |
| anisotropic_relative_rotation | {'rs0': -0.25, 'rp0': 0.35, 'mixing0': 0.05, 'toy_rotation_only': True} |

## 4. Integration formula

`F_toy/A = k_B*T*sum_n' integral Q dQ dphi/(2*pi)^2 logdet[I-exp(-2*kappa*d) R1_toy R2_toy]`。这个公式只用于 toy matrices。

## 5. Baseline toy results

| quantity | value |
| --- | --- |
| temperature_K | 10.0 |
| separation_nm | 100.0 |
| Qc_nm_inv | 0.2 |
| omega_c_eV | 0.05 |

## 6. Zero and isotropic checks

| check | status |
| --- | --- |
| zero_toy_integration | PASS |
| isotropic_angle_independence | PASS |

## 7. Anisotropic toy angle checks

| check | status |
| --- | --- |
| anisotropic_toy_periodicity | PASS |

## 8. Distance dependence

| check | status |
| --- | --- |
| distance_dependence | PASS |

## 9. Convergence scans

| scan | values | F_toy/A | relative_changes | status |
| --- | --- | --- | --- | --- |
| n_max | [2, 4, 8] | [-1.652763814387433e-12, -2.5303746169448495e-12, -3.6923045249569564e-12] | [None, 0.5309958960365351, 0.45919284055062576] | PASS |
| Q_max | [250000000.0, 500000000.0, 750000000.0] | [-9.584195651604788e-11, -3.6923045249569564e-12, -7.749333241351442e-14] | [None, 0.9614750714699912, 0.9790122044675018] | PASS |
| n_Q | [8, 12, 16] | [-4.343291118123455e-14, -3.6923045249569564e-12, -2.3569261594417926e-11] | [None, 84.01167489210897, 5.3833471576109195] | PASS |
| n_phi | [8, 12, 16] | [-3.692304524956956e-12, -3.6923045249569564e-12, -3.6923045249569564e-12] | [None, 2.1877761205401476e-16, 0.0] | PASS |

## 10. Imaginary-part sanity

| check | status |
| --- | --- |
| imaginary_part_sanity | PASS |

## 11. What this is not

- Toy-model full integration is not a real material Casimir energy calculation.
- No LNO327 material response grid is used.
- Do not interpret toy energy density as physical prediction.
- Next real-material stage requires R_TE_TM(i*xi_n,Q,phi) or sigma_tilde(i*xi_n,Q,phi) on a production grid.

## 12. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_10_status | STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED |
| recommended_next_action | Proceed to real-material response/reflection grid generation planning; still do not run production torque. |

## 13. Recommended next step

Proceed to real-material response/reflection grid generation planning; still do not run production torque.
