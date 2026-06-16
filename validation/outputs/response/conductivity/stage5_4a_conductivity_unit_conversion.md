# Stage 5.4a 电导单位转换验证

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

## 2. Input conductivity convention

| quantity | value |
| --- | --- |
| input | sigma_model_ij |
| input_formula | sigma_model_ij(iOmega) = - Pi_ij(iOmega) / omega_eV |
| output_si | sigma_SI_sheet_ij |
| output_dimensionless | sigma_tilde_ij = Z0 * sigma_SI_sheet_ij |
| normalization | bilayer-normalized 2D sheet conductivity |
| bulk_3d_conductivity | False |
| single_layer_conductivity | False |

## 3. Analytic unit-chain derivation

$A_i^{model}=ea_iA_i^{SI}/\hbar$，因此 $\sigma^{SI,sheet}_{ij}=(e^2/\hbar)(a_i a_j/A_{cell})\sigma^{model}_{ij}$。

## 4. Geometry tensor

| quantity | value |
| --- | --- |
| material_lattice_config | LNO327_thin_film_SrLaAlO4_clamped |
| is_placeholder | False |
| source_note | Default in-plane lattice constant for coherently strained thin-film LNO327 / (La,Pr)327-type films on SrLaAlO4-like substrate. Use as a thin-film working value, not as relaxed bulk La3Ni2O7. |
| lattice_a_x_m | 3.754e-10 |
| lattice_a_y_m | 3.754e-10 |
| unit_cell_area_m2 | 1.4092516e-19 |
| geometry_tensor | [[1. 1.]
 [1. 1.]] |
| lattice_note | 3.754 Angstrom is the current thin-film working default; sample-specific constants may override this config. |

## 5. SI sheet conductivity conversion

这是 one-bilayer sheet response，不是 bulk 3D conductivity，也不是 single-layer conductivity。

## 6. Dimensionless sheet conductivity

$\tilde\sigma_{ij}=Z_0\sigma^{SI,sheet}_{ij}$。$\tilde\sigma$ 不是新的材料模型参数，而是 dimensionless sheet conductivity / dimensionless sheet admittance；不再使用 $g$ 作为符号。

## 7. Synthetic square-lattice check

status: PASS

## 8. Synthetic rectangular-lattice check

status: PASS

## 9. Dimensionless prefactor check

| quantity | value |
| --- | --- |
| Z0 e^2/hbar | 0.09170123682716239 |
| 4 pi alpha | 0.09170123682663807 |
| abs error | 5.24316701167038e-13 |
| status | PASS |

## 10. Diagnostic decision

| quantity | value |
| --- | --- |
| stage5_4a_status | STAGE5_4A_CONDUCTIVITY_UNIT_CONVERSION_PASSED |
| recommended_next_action | Proceed to Stage 5.4b to convert validated model conductivity outputs to SI sheet and sigma_tilde data; still do not run reflection/Casimir. |

## 11. Recommended next step

Proceed to Stage 5.4b to convert validated model conductivity outputs to SI sheet and sigma_tilde data; still do not run reflection/Casimir. 本阶段仍未进入 reflection/Casimir，也尚未做真实材料 lattice constants 的最终配置管理。
