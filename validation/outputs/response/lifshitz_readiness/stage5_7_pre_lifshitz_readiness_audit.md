# Stage 5.7 pre-Lifshitz readiness audit

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
- no_lifshitz_trace_log_production: True
- no_casimir_energy: True
- no_casimir_force: True
- no_casimir_torque: True
- not_casimir_ready_claim: True

## 2. Input source

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/reflection_input/stage5_6_te_tm_reflection_adapter.json |
| input_stage | Stage 5.6 |
| input_status | STAGE5_6_TE_TM_ADAPTER_PASSED |
| num_input_cases | 8 |

## 3. Matrix convention

`R^{TE/TM}` 使用 TE/TM amplitude basis，顺序为 `['s', 'p']`。行是 reflected polarization，列是 incident polarization，定义为 `E_ref = R E_inc`。

## 4. Trace-log integrand convention

`M = I - exp(-2*kappa*d) * R1 @ R2`，单点 integrand 是 `log(det(M))`。`R1` 和 `R2` 必须在同一个 lab-frame TE/TM basis 中表达。本阶段只检查 integrand-level object。

## 5. Plate rotation convention

材料旋转角 `theta` 是 plate 2 crystal axes 相对 plate 1/lab axes 的旋转角。在材料自己的 crystal frame 中，`Q_crystal = R(-theta) Q_lab`。最终 `R_TE_TM` 都必须回到共同 lab TE/TM basis。

## 6. Synthetic checks

| check | status |
| --- | --- |
| zero_reflection | PASS |
| large_separation | PASS |
| zero_sheet | PASS |
| isotropic_identical_sheets | PASS |
| isotropic_angle_independence | PASS |
| rotation_convention | PASS |
| matrix_order | PASS |

## 7. Representative real Stage 5.6 integrand-level checks

| q | n | exp(-2*kappa*d) | logdet |
| --- | --- | --- | --- |
| q_diag_pos | 1 | 3.0254840046016336e-06 | (-3.008121903138557e-06-9.297157994270533e-25j) |
| q_diag_neg | 1 | 3.0254840046016336e-06 | (-3.008121903138557e-06-3.064320794198276e-24j) |
| q_diag_pos | 2 | 3.025387216610241e-06 | (-2.9846365860204107e-06+1.0323319370247204e-23j) |
| q_diag_neg | 2 | 3.025387216610241e-06 | (-2.9846365860204107e-06-5.157214210016617e-25j) |
| q_diag_pos | 4 | 3.0250000980431622e-06 | (-2.8975512343153922e-06-4.519949425091268e-25j) |
| q_diag_neg | 4 | 3.0250000980431622e-06 | (-2.8975512343153922e-06-4.58439149390064e-24j) |

这些数值只来自 identical-sheet toy pair，不是完整 Matsubara sum，也不是完整 d^2Q 积分。

## 8. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_7_status | STAGE5_7_PRE_LIFSHITZ_READINESS_PASSED |
| recommended_next_action | Proceed to Casimir integrand prototype with controlled synthetic/material-grid inputs; still do not run production torque. |

| summary | value |
| --- | --- |
| separation_m | 1.0000000000000001e-07 |
| num_representative_rows | 6 |
| max_abs_representative_logdet | 3.008121903138557e-06 |
| max_abs_representative_imag_logdet | 1.0323319370247204e-23 |

## 9. Recommended next step

Proceed to Casimir integrand prototype with controlled synthetic/material-grid inputs; still do not run production torque. 当前没有输出 Casimir energy、force 或 torque，也没有声明 production-ready。
