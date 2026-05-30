# response 输出入口

本目录保存 response 层诊断输出。大型 `.csv`、`.npz`、`.png` 是复现数据和图像，
不是主要阅读入口。当前推荐先读 `outputs/phase_reports/` 下的阶段报告。

## 当前 finite-q 主线

- `finite_q_anisotropy/`：A4 角向各向异性初步诊断；由于 small-q continuity 未整体通过，
  不能做物理解释。
- `finite_q_local_limit/`：finite-q bubble 的 local-limit decomposition。
- `finite_q_formula_consistency/`：vertex、BZ wrapping、denominator、overlap 初排查。
- `finite_q_subspace_repair/`：projector / denominator repair 诊断。
- `finite_q_raw_q0_consistency/`：raw q=0 bubble 与 local components 的定义层级诊断。

## 历史和辅助诊断

`bdg_normal_limit/`、`convergence_imag/`、`high_nk_convergence/`、
`local_sheet_imag/`、`static_policy_comparison/`、`n0_sensitivity/` 等目录保留为
response 层历史稳定性、接口边界或 policy 诊断。它们不应被当作当前 finite-q 主线的
最终结论。

## 推荐阅读顺序

1. `outputs/phase_reports/current_project_status.md`
2. `outputs/phase_reports/finite_q_response_status.md`
3. `outputs/response/finite_q_subspace_repair/finite_q_subspace_repair_summary.md`
4. `outputs/response/finite_q_raw_q0_consistency/finite_q_raw_q0_consistency_summary.md`

## 维护原则

- 不删除旧 outputs。
- 不使用 `.gitignore` 隐藏结果。
- 不把 response diagnostic 输出解释为正式 Casimir 结果。
