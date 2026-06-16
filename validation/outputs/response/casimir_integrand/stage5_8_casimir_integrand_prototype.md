# Stage 5.8 Casimir trace-log integrand prototype

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
| input_json | validation/outputs/response/reflection_input/stage5_6_te_tm_reflection_adapter.json |
| input_stage | Stage 5.6 |
| input_status | STAGE5_6_TE_TM_ADAPTER_PASSED |
| num_input_cases | 8 |

## 3. Integrand convention

`M = I - exp(-2*kappa*d) * R1 @ R2`，prototype value 是 `log(det(M))`。`R1` 和 `R2` 使用 TE/TM amplitude basis，顺序为 `['s', 'p']`，行是 reflected polarization，列是 incident polarization。

## 4. Prototype scope

| scope | enabled |
| --- | --- |
| full_matsubara_sum | False |
| full_Q_integral | False |
| casimir_energy | False |
| casimir_force | False |
| casimir_torque | False |
| production_run | False |
| toy_rotation_only_not_physical_material_rotation | True |

## 5. Synthetic checks

| check | status |
| --- | --- |
| zero_reflection | PASS |
| one_zero_plate | PASS |
| large_separation | PASS |
| small_separation_magnitude | PASS |
| isotropic_identical_sheets_formula | PASS |
| isotropic_angle_independence | PASS |
| anisotropic_toy_periodicity | PASS |
| matrix_order | PASS |

## 6. Representative Stage 5.6 integrand-level values

| q | n | exp(-2*kappa*d) | logdet |
| --- | --- | --- | --- |
| q_diag_pos | 1 | 3.0254840046016336e-06 | (-3.008121903138557e-06-9.297157994270533e-25j) |
| q_diag_neg | 1 | 3.0254840046016336e-06 | (-3.008121903138557e-06-3.064320794198276e-24j) |
| q_diag_pos | 2 | 3.025387216610241e-06 | (-2.9846365860204107e-06+1.0323319370247204e-23j) |
| q_diag_neg | 2 | 3.025387216610241e-06 | (-2.9846365860204107e-06-5.157214210016617e-25j) |
| q_diag_pos | 4 | 3.0250000980431622e-06 | (-2.8975512343153922e-06-4.519949425091268e-25j) |
| q_diag_neg | 4 | 3.0250000980431622e-06 | (-2.8975512343153922e-06-4.58439149390064e-24j) |

这些 representative material rows 只是 validation-point integrand-level checks，不是物理能量或力矩。

## 7. What this is not

本阶段没有 full Matsubara sum，没有 full Q integral，没有输出 Casimir energy、force 或 torque。anisotropic toy periodicity 只是 synthetic matrix check，不是 LNO327 物理 torque。

## 8. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_8_status | STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED |
| recommended_next_action | Proceed to material response grid planning for full Matsubara/Q integration; do not run production torque yet. |

| summary | value |
| --- | --- |
| separation_m | 1.0000000000000001e-07 |
| num_representative_rows | 6 |
| max_abs_representative_logdet | 3.008121903138557e-06 |
| max_abs_representative_imag_logdet | 1.0323319370247204e-23 |
| synthetic_checks_all_pass | True |

## 9. Recommended next step

Proceed to material response grid planning for full Matsubara/Q integration; do not run production torque yet.
