# Stage 5.6 TE/TM reflection adapter

## 1. Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_heavy_response_run: True
- no_lifshitz_trace_log: True
- no_casimir_energy: True
- no_casimir_force: True
- no_casimir_torque: True
- not_casimir_ready_claim: True

## 2. Input source

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/reflection_input/stage5_5b_reflection_input_tensor.json |
| input_stage | Stage 5.5b |
| input_status | STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED |
| num_input_cases | 8 |

## 3. Why TE/TM adapter is needed

Stage 5.5b 的 `R_E^{LT}` 是内部 tangential electric basis 下的反射输入矩阵。标准 Lifshitz/Casimir trace-log 通常使用 TE/TM amplitude basis；本阶段只做基底适配，当前没有计算 trace-log。

## 4. Internal (L/T) basis

`L` 平行于面内 SI 波矢 `Q`，`T = z_hat cross L`。内部顺序是 `['L', 'T']`，并保留 `sigma_tilde_LT` 和 `R_E_LT` 作为审计量。

## 5. TE/TM amplitude convention

输出顺序是 `['s', 'p']`，其中 `s/TE` 对应 `T` 方向电场，`p/TM` 对应 `L-z` 平面内电场。本文采用 `E_s = E_T`、`E_p_inc = E_L_inc`、`E_p_ref = -E_L_ref`。`p` 反射振幅的负号是本 adapter convention 的一部分。

## 6. Adapter formula

`R_TE_TM = [[R_TT, R_TL], [-R_LT, -R_LL]]`。这里 `R_E_LT` 的行列顺序为 `(L,T)`，`R_TE_TM` 的行列顺序为 `(s,p)`。

## 7. Synthetic checks

| check | status | classification |
| --- | --- | --- |
| adapter_index_check | PASS |  |
| zero_sheet_check | PASS |  |
| isotropic_scalar_sheet_no_mixing | PASS |  |
| te_tm_scalar_limit_consistency | PASS |  |
| strong_sheet_limit | PASS |  |
| weak_sheet_limit | PASS |  |
| symmetric_offdiag_mixing | PASS | symmetric_finite_q_mixing |
| hall_like_antisymmetric_marker | PASS | antisymmetric_marker |

## 8. Representative R_TE_TM rows

| q | n | R_TE_TM | adapter_delta_max |
| --- | --- | --- | --- |
| q_diag_pos | 1 | [[-8.40458675e-04-1.89875396e-20j -1.84751771e-07+4.69857766e-22j]
 [ 1.10087494e-01-2.87185200e-16j  9.97125478e-01+1.53968832e-19j]] | 0.0 |
| q_diag_neg | 1 | [[-8.40458675e-04-7.09351056e-21j  1.84751771e-07-4.54630416e-22j]
 [-1.10087494e-01+3.09879570e-16j  9.97125478e-01+5.07763068e-19j]] | 0.0 |
| q_diag_pos | 2 | [[-1.14716334e-03+3.04740243e-19j -2.80136028e-07-3.14890105e-22j]
 [ 4.17311677e-02+5.64810693e-17j  9.93240996e-01-1.71733925e-18j]] | 0.0 |
| q_diag_neg | 2 | [[-1.14716334e-03-4.32161451e-20j  2.80136028e-07-7.32578068e-22j]
 [-4.17311677e-02+1.06513071e-16j  9.93240996e-01+8.57013157e-20j]] | 0.0 |
| q_diag_pos | 4 | [[-1.29171860e-03+1.80623645e-20j -3.11830131e-07+1.40697473e-21j]
 [ 1.16133737e-02-5.30527524e-17j  9.78705827e-01+7.63254240e-20j]] | 0.0 |
| q_diag_neg | 4 | [[-1.29171860e-03-4.74754270e-20j  3.11830131e-07-5.88171432e-22j]
 [-1.16133737e-02+2.15608348e-17j  9.78705827e-01+7.74158629e-19j]] | 0.0 |

## 9. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_6_status | STAGE5_6_TE_TM_ADAPTER_PASSED |
| recommended_next_action | Proceed to trace-log integrand convention audit; still do not compute full Casimir energy/torque. |

| summary | value |
| --- | --- |
| num_cases | 8 |
| max_abs_R_TE_TM | 0.9971254782566219 |
| max_abs_R_TE_TM_offdiag | 0.11008749397653299 |
| q_sign_offdiag_consistency | PASS_adapter_formula_and_representative_q_diag_offdiag_flip |
| all_cases_converted | True |
| all_adapter_formula_deltas_small | True |

## 10. Recommended next step

Proceed to trace-log integrand convention audit; still do not compute full Casimir energy/torque. 本阶段没有修改 response、conductivity convention、bubble sign、direct contact、source/observable、Ward convention；没有 fitted contact，没有 `E^{ET}`，没有 heavy response，没有 Lifshitz trace-log，没有 Casimir energy/force/torque。文档和 metadata 始终使用 `sigma_tilde`，不把它另记为其他裸符号。
