# Stage 5.13 zero-mode and grid-convergence planning audit

## Boundary

- no_response_formula_change: True
- no_main_response_change: True
- no_bubble_sign_change: True
- no_direct_contact_change: True
- no_source_observable_change: True
- no_residual_tuning: True
- no_fitted_contact: True
- no_E_ET_added: True
- no_conductivity_unit_change: True
- no_reflection_convention_change: True
- no_trace_log_convention_change: True
- no_production_energy: True
- no_force_output: True
- no_torque_output: True
- not_casimir_ready_claim: True

## Input

| quantity | value |
| --- | --- |
| input_json | validation/outputs/response/material_energy_prototype/stage5_12_small_real_material_energy_prototype.json |
| input_stage | Stage 5.12 |
| input_status | STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_PASSED |

## Small-Q Audit

| quantity | value |
| --- | --- |
| num_points | 48 |
| num_completed | 48 |
| num_failed | 0 |
| max_ward_residual | 6.270616185686254e-07 |
| max_abs_sigma_tilde | 6.152490748649776 |
| max_abs_R_TE_TM | 0.9990615847502894 |
| max_abs_logdet | 0.45656181433593196 |
| smoothness | {'max_relative_jump': 0.0, 'num_jumps': 0} |

## Zero-Mode Audit

| quantity | value |
| --- | --- |
| num_points | 36 |
| num_completed | 36 |
| num_failed | 0 |
| max_ward_residual | 4.7051517545933075e-07 |
| max_abs_sigma_tilde | 106.86515204213161 |
| max_abs_R_TE_TM | 0.9999224838462286 |
| max_abs_logdet | 0.14538920162723065 |
| smoothness | {'max_relative_jump': 0.0, 'num_jumps': 0} |

## Grid Convergence Plan

{
  "coarse": {
    "n_max": 8,
    "n_Q": 16,
    "n_phi": 8
  },
  "medium": {
    "n_max": 16,
    "n_Q": 24,
    "n_phi": 12
  },
  "fine": {
    "n_max": 32,
    "n_Q": 32,
    "n_phi": 16
  },
  "Q0_policy": "exclude endpoint Q=0 and use interior quadrature nodes",
  "n0_policy": "use extrapolated xi->0+ reflection matrix; do not divide by omega=0",
  "Q_max_convergence": "scan Q_max separately",
  "n_max_convergence": "scan n_max separately",
  "angular_radial_convergence": "scan n_Q and n_phi separately",
  "response_grid_strategy": "start with direct response grid for audit, then evaluate interpolation grid only after error controls exist"
}

## Checks

| check | status |
| --- | --- |
| input_status | PASS |
| small_Q_audit | PASS |
| zero_mode_audit | PASS |
| grid_plan_present | PASS |
| no_production_energy | PASS |
| no_force_torque | PASS |

## Notes

Q=0 不作为普通点；Q->0+ 应使用内部 quadrature 节点。n=0 不能直接使用 sigma=-Pi/Omega，应由 xi->0+ 的 R_TE_TM 极限获得。本阶段不输出 production energy、force 或 torque。

## Diagnostic Decision

| quantity | value |
| --- | --- |
| stage5_13_status | STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED |
| recommended_next_action | Proceed only to production-grid convergence run after zero-mode and Q->0 handling are accepted. |
