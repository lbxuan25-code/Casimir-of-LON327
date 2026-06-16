# Stage 5.4b SI sheet / sigma_tilde 转换

## 1. Boundary

- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_reflection_casimir: True
- not_casimir_ready_claim: True
- no_heavy_response_run: True

## 2. Input file and input status

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/conductivity/stage5_2_bilayer_sheet_conductivity_sanity_scan.json |
| input_stage | Stage 5.2 |
| input_diagnostic_status | CONDUCTIVITY_SANITY_MONITOR_OFFDIAG |
| input_num_cases | 8 |
| allow_monitor_input | True |
| require_no_fail_input | True |

## 3. Conductivity convention

| quantity | value |
| --- | --- |
| input | sigma_model_ij |
| model_formula | sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV |
| output_si | sigma_SI_sheet_ij |
| output_dimensionless | sigma_tilde_ij = Z0 * sigma_SI_sheet_ij |
| normalization | bilayer-normalized 2D sheet conductivity |
| bulk_3d_conductivity | False |
| single_layer_conductivity | False |

## 4. Lattice convention

| quantity | value |
| --- | --- |
| name | LNO327_thin_film_SrLaAlO4_clamped |
| lattice_a_x_m | 3.754e-10 |
| lattice_a_y_m | 3.754e-10 |
| unit_cell_area_m2 | 1.4092516e-19 |
| source_note | Default in-plane lattice constant for coherently strained thin-film LNO327 / (La,Pr)327-type films on SrLaAlO4-like substrate. Use as a thin-film working value, not as relaxed bulk La3Ni2O7. |
| is_placeholder | False |

## 5. Unit conversion formula

$\sigma^{SI,sheet}_{ij}=(e^2/\hbar)(a_i a_j/A_{cell})\sigma^{model}_{ij}$，$\tilde\sigma_{ij}=Z_0\sigma^{SI,sheet}_{ij}$。

## 6. Converted conductivity summary

| quantity | value |
| --- | --- |
| num_cases | 8 |
| max_abs_sigma_tilde | 1.2783151986587722 |
| max_abs_sigma_SI_sheet_S | 0.003393183805893206 |
| min_diag_sigma_SI_sheet_real_S | 0.0006820179748403371 |
| max_relative_offdiag_norm_model | 0.13186592735850317 |
| max_relative_offdiag_norm_tilde | 0.13186592735850317 |
| conversion_preserves_relative_structure | True |

## 7. Representative converted values

| q | n | scale | sigma_tilde_xx | sigma_tilde_xy | sigma_tilde_yy |
| --- | --- | --- | --- | --- | --- |
| q_diag_pos | 1 | 1.0 | (0.9300168470299763+2.549975674921206e-16j) | (-0.14740172145326583-7.083071377034296e-17j) | (1.2783151986587684-2.3317380040664484e-16j) |
| q_diag_neg | 1 | 1.0 | (0.93001684702998+3.219651761379651e-16j) | (0.14740172145326602+2.3592057784091e-17j) | (1.2783151986587722-2.076048652101459e-16j) |
| q_diag_pos | 2 | 1.0 | (0.769748591130418-2.348956755357977e-16j) | (-0.04475551162211444+3.384327369806995e-17j) | (0.879609269221585-1.926984588235845e-16j) |

## 8. Model vs SI vs sigma_tilde

`sigma_tilde` 是 dimensionless sheet conductivity / admittance，不再使用 `g` 作为符号。这是 one-bilayer sheet response，不是 bulk 3D，也不是 single-layer。

## 9. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_4b_status | STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED |
| recommended_next_action | Proceed to reflection-input preparation only after checking tensor formatting; do not run reflection/Casimir yet. |

## 10. Recommended next step

Proceed to reflection-input preparation only after checking tensor formatting; do not run reflection/Casimir yet. 本阶段没有运行 heavy response，也没有进入 reflection/Casimir。LNO327 thin-film lattice constant is now 3.754 Å, not placeholder 3.85 Å; future sample-specific constants may override the config.
