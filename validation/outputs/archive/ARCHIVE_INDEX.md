# Validation Archive Index

本文件记录验证层历史结果和脚本的位置。主 `outputs/` 现在只保留材料和模型本征结果；
收敛性、诊断、benchmark-only 输出集中在 `validation/`。

## Current Validation Entries

- `validation/README.md`
- `validation/outputs/response/finite_q_raw_q0_consistency/`
- `validation/outputs/casimir/local_response_integral/distance_scan/`
- `validation/outputs/numerical_stability/README.md`
- `validation/cache/casimir_local_response/response_tensors/`

## Archived Outputs

- `validation/outputs/archive/normal_state/sampling_convergence/`
- `validation/outputs/archive/normal_state/fs_sensitive_sampling/`
- `validation/outputs/archive/normal_state/fs_adaptive_integration/`
- `validation/outputs/archive/response/bdg_normal_limit/`
- `validation/outputs/archive/response/convergence_imag/`
- `validation/outputs/archive/response/high_nk_convergence/`
- `validation/outputs/archive/response/local_sheet_imag/`
- `validation/outputs/archive/response/n0_sensitivity/`
- `validation/outputs/archive/response/nonlocal_interface/`
- `validation/outputs/archive/response/static_policy_comparison/`
- `validation/outputs/archive/response/static_response/`
- `validation/outputs/archive/response/unit_audit/`
- `validation/outputs/archive/response/finite_q_anisotropy/`
- `validation/outputs/archive/response/finite_q_formula_consistency/`
- `validation/outputs/archive/response/finite_q_local_limit/`
- `validation/outputs/archive/response/finite_q_subspace_repair/`
- `validation/outputs/archive/casimir/local_response_integral/`
- `validation/outputs/archive/smoke/smoke/`

## Validation Scripts

- `validation/scripts/numerical_stability/`
- `validation/scripts/finite_q_diagnostics/`
- `validation/scripts/response/`
- `validation/scripts/casimir/`
- `validation/scripts/smoke/`
- `validation/scripts/compat/`

## Historical Move Summary

- Former `scripts/archive/numerical_stability/` scripts now live in
  `validation/scripts/numerical_stability/`.
- Former `scripts/archive/finite_q_diagnostics/` scripts now live in
  `validation/scripts/finite_q_diagnostics/`.
- Former `scripts/archive/legacy_smoke/` scripts now live in
  `validation/scripts/smoke/`.
- Former `scripts/response/` diagnostic scripts now live in
  `validation/scripts/response/`.
- Former `scripts/casimir/` benchmark scripts now live in
  `validation/scripts/casimir/`.
- Former `outputs/archive/` now lives in `validation/outputs/archive/`.
- Former `outputs/response/` finite-q diagnostic output now lives in
  `validation/outputs/response/`.
- Former `outputs/casimir/` benchmark output now lives in
  `validation/outputs/casimir/`.
- Former `outputs/cache/` now lives in `validation/cache/`.

## Reading Rule

For current material properties, start from `outputs/`. For numerical credibility,
formula diagnostics, and benchmark evidence, start from `validation/README.md`.
