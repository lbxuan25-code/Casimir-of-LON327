# Stage 5.5b reflection-input tensor formatter

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
| input_json | validation/outputs/response/conductivity/stage5_4b_si_sheet_dimensionless_conductivity.json |
| input_stage | Stage 5.4b |
| input_status | STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED |
| num_input_cases | 8 |

## 3. (L/T) basis definition

(L/T) 不是全局固定坐标，而是每个 $\mathbf q$ 点上的局部坐标：$L\parallel\mathbf Q$，$T=\hat z\times L$。

## 4. Frequency and wave-vector conversion

| quantity | value |
| --- | --- |
| q_model_to_SI | Q_x = q_model_x/a_x, Q_y = q_model_y/a_y |
| omega_eV_to_xi | xi = omega_eV * eV_J / hbar |
| kappa | sqrt(Q^2 + xi^2/c^2) |

## 5. sigma_tilde_xy to sigma_tilde_LT

sigma_tilde_LT = R_Q sigma_tilde_xy R_Q^T

## 6. Vacuum admittance Y0

Y0_LT = diag(xi/(c*kappa), c*kappa/xi)

## 7. Tangential electric reflection-input matrix

R_E_LT = - solve(2*Y0_LT + sigma_tilde_LT, sigma_tilde_LT)。这是 tangential electric field reflection-input matrix，尚未转换成文献 TE/TM amplitude convention。

## 8. Synthetic checks

| check | status |
| --- | --- |
| qx_basis_check | PASS |
| qy_basis_sign_check | PASS |
| isotropic_scalar_sheet_check | PASS |
| weak_sheet_limit | PASS |
| diagonal_LT_no_mixing | PASS |
| offdiag_LT_retains_mixing | PASS |
| hall_like_antisymmetric_marker | PASS |

## 9. Representative formatted rows

| q | n | Q | kappa | R_E_LT |
| --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 63542144.069667354 | 63542197.388795726 | [[-9.97125478e-01-1.53968832e-19j -1.10087494e-01+2.87185200e-16j]
 [-1.84751771e-07+4.69857766e-22j -8.40458675e-04-1.89875396e-20j]] |
| q_diag_neg | 1 | 63542144.069667354 | 63542197.388795726 | [[-9.97125478e-01-5.07763068e-19j  1.10087494e-01-3.09879570e-16j]
 [ 1.84751771e-07-4.54630416e-22j -8.40458675e-04-7.09351056e-21j]] |
| q_diag_pos | 2 | 63542144.069667354 | 63542357.34591241 | [[-9.93240996e-01+1.71733925e-18j -4.17311677e-02-5.64810693e-17j]
 [-2.80136028e-07-3.14890105e-22j -1.14716334e-03+3.04740243e-19j]] |
| q_diag_neg | 2 | 63542144.069667354 | 63542357.34591241 | [[-9.93240996e-01-8.57013157e-20j  4.17311677e-02-1.06513071e-16j]
 [ 2.80136028e-07-7.32578068e-22j -1.14716334e-03-4.32161451e-20j]] |
| q_diag_pos | 4 | 63542144.069667354 | 63542997.17035253 | [[-9.78705827e-01-7.63254240e-20j -1.16133737e-02+5.30527524e-17j]
 [-3.11830131e-07+1.40697473e-21j -1.29171860e-03+1.80623645e-20j]] |
| q_diag_neg | 4 | 63542144.069667354 | 63542997.17035253 | [[-9.78705827e-01-7.74158629e-19j  1.16133737e-02-2.15608348e-17j]
 [ 3.11830131e-07-5.88171432e-22j -1.29171860e-03-4.74754270e-20j]] |

## 10. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_5b_status | STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED |
| recommended_next_action | Proceed to TE/TM adapter convention audit; still do not compute Casimir energy/torque. |

## 11. Recommended next step

Proceed to TE/TM adapter convention audit; still do not compute Casimir energy/torque. 尚未计算 Lifshitz trace-log，也尚未计算 Casimir energy/force/torque。
