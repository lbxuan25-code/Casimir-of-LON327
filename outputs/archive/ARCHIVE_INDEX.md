# 归档清单

本文件记录本轮工程归档整理的移动路径。所有移动都保留历史文件，不删除结果，
不使用 `.gitignore`，不改变任何物理公式。

## active 目录

当前仍作为主入口保留：

- `docs/reports/`
- `outputs/response/finite_q_raw_q0_consistency/`
- `outputs/casimir/local_response_integral/distance_scan/`
- `outputs/cache/casimir_local_response/response_tensors/`
- `docs/notes/numerical_stability_summary.md`
- `README.md`
- `outputs/README.md`
- `scripts/benchmark_casimir_local_response_distance_scan.py`
- `scripts/diagnose_finite_q_raw_q0_consistency.py`
- `scripts/benchmark_casimir_local_response_integral.py`
- `scripts/refine_casimir_local_convergence_blockers.py`
- `scripts/run_casimir_local_convergence_final.py`
- `scripts/converge_casimir_local_response_integral.py`

## response outputs

- `outputs/archive/response/bdg_normal_limit/` -> `outputs/archive/response/bdg_normal_limit/`
- `outputs/archive/response/convergence_imag/` -> `outputs/archive/response/convergence_imag/`
- `outputs/archive/response/high_nk_convergence/` -> `outputs/archive/response/high_nk_convergence/`
- `outputs/archive/response/local_sheet_imag/` -> `outputs/archive/response/local_sheet_imag/`
- `outputs/archive/response/n0_sensitivity/` -> `outputs/archive/response/n0_sensitivity/`
- `outputs/archive/response/nonlocal_interface/` -> `outputs/archive/response/nonlocal_interface/`
- `outputs/archive/response/static_policy_comparison/` -> `outputs/archive/response/static_policy_comparison/`
- `outputs/archive/response/static_response/` -> `outputs/archive/response/static_response/`
- `outputs/archive/response/unit_audit/` -> `outputs/archive/response/unit_audit/`
- `outputs/response/finite_q_anisotropy/` -> `outputs/archive/response/finite_q_anisotropy/`
- `outputs/response/finite_q_formula_consistency/` -> `outputs/archive/response/finite_q_formula_consistency/`
- `outputs/response/finite_q_local_limit/` -> `outputs/archive/response/finite_q_local_limit/`
- `outputs/response/finite_q_subspace_repair/` -> `outputs/archive/response/finite_q_subspace_repair/`

这些目录用于历史 response 稳定性、接口边界和旧 finite-q 诊断追溯。

## normal_state outputs

- `outputs/archive/normal_state/sampling_convergence/` -> `outputs/archive/normal_state/sampling_convergence/`
- `outputs/archive/normal_state/fs_sensitive_sampling/` -> `outputs/archive/normal_state/fs_sensitive_sampling/`
- `outputs/archive/normal_state/fs_adaptive_integration/` -> `outputs/archive/normal_state/fs_adaptive_integration/`

这些目录用于追溯 normal-state sampling 和 FS-adaptive 数值稳定性证据。

## smoke outputs

- `outputs/smoke/` -> `outputs/archive/smoke/smoke/`

这些目录只用于历史 smoke / 接口连通性检查。

## casimir outputs

- `outputs/casimir/local_response_integral/final_convergence/` -> `outputs/archive/casimir/local_response_integral/final_convergence/`
- `outputs/casimir/local_response_integral/refined_convergence/` -> `outputs/archive/casimir/local_response_integral/refined_convergence/`
- `outputs/archive/casimir/local_response_integral/data/` -> `outputs/archive/casimir/local_response_integral/data/`
- `outputs/archive/casimir/local_response_integral/convergence/` -> `outputs/archive/casimir/local_response_integral/convergence/`
- `outputs/casimir/local_response_integral/figures/` -> `outputs/archive/casimir/local_response_integral/figures/`
- `outputs/casimir/data/` -> `outputs/archive/casimir/data/`
- `outputs/casimir/figures/` -> `outputs/archive/casimir/figures/`

- `outputs/casimir/local_response_integral/cache/` -> `outputs/cache/casimir_local_response/response_tensors/`

cache 单独放在 `outputs/cache/`，避免把可复用中间张量和某一次 distance scan 输出混在一起。

## scripts

### numerical_stability

- `scripts/benchmark_bdg_normal_limit.py` -> `scripts/archive/numerical_stability/benchmark_bdg_normal_limit.py`
- `scripts/convergence_response_imag.py` -> `scripts/archive/numerical_stability/convergence_response_imag.py`
- `scripts/refine_high_nk_convergence.py` -> `scripts/archive/numerical_stability/refine_high_nk_convergence.py`
- `scripts/diagnose_normal_sampling_convergence.py` -> `scripts/archive/numerical_stability/diagnose_normal_sampling_convergence.py`
- `scripts/benchmark_normal_fs_sensitive_sampling.py` -> `scripts/archive/numerical_stability/benchmark_normal_fs_sensitive_sampling.py`
- `scripts/benchmark_normal_fs_adaptive_integration.py` -> `scripts/archive/numerical_stability/benchmark_normal_fs_adaptive_integration.py`
- `scripts/assess_n0_torque_sensitivity.py` -> `scripts/archive/numerical_stability/assess_n0_torque_sensitivity.py`

### finite_q_diagnostics

- `scripts/diagnose_finite_q_response_anisotropy.py` -> `scripts/archive/finite_q_diagnostics/diagnose_finite_q_response_anisotropy.py`
- `scripts/diagnose_finite_q_local_limit_decomposition.py` -> `scripts/archive/finite_q_diagnostics/diagnose_finite_q_local_limit_decomposition.py`
- `scripts/diagnose_finite_q_formula_consistency.py` -> `scripts/archive/finite_q_diagnostics/diagnose_finite_q_formula_consistency.py`
- `scripts/diagnose_finite_q_subspace_denominator_repair.py` -> `scripts/archive/finite_q_diagnostics/diagnose_finite_q_subspace_denominator_repair.py`

### legacy_smoke

- `scripts/smoke_casimir_local_response.py` -> `scripts/archive/legacy_smoke/smoke_casimir_local_response.py`

## 查看或恢复历史结果

查看历史结果时，从本文件找到新路径，再进入对应 archive 目录读取 summary 或数据。
如果未来确需恢复某个脚本或输出目录，应使用 Git 移动记录或反向 `git mv`，并同步更新
README 和测试路径。
