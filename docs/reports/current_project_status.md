# 当前项目总状态

## 总览

local-response 数值稳定性阶段已经完成归纳。normal response 的 FS-adaptive sampling、
BdG response 基础稳定性、local-response Casimir integral convergence、refined cutoff /
Matsubara scan 以及 zero-torque baseline 均已形成 benchmark 级别证据。

local-response distance scan benchmark 已完成。normal、spm、dwave 三个 kind 在当前
local-response 设置下均保持 zero-torque baseline。

当前主线已经转入 finite-q response 层诊断。该主线的目标是在不修改 H0、不修改 pairing
的前提下，检查有限 `q_parallel` 是否能够在 response 层放大 spm/dwave 角向差异。

本轮仓库结构已经完成 docs / scripts / outputs 分层整理。当前 `docs/reports/` 保留阶段报告，
`outputs/` 只保留 active 数据产物、cache 和 archive；旧 response、normal sampling、
smoke 和中间 Casimir convergence 输出已移动到 `validation/outputs/archive/`，移动清单见
`validation/outputs/archive/ARCHIVE_INDEX.md`。

## 当前边界

finite-q response 仍处于 diagnostic prototype 阶段：

- `gauge_status=prototype_not_ward_verified`
- `final_casimir_input=False`
- `not_final_Casimir_conclusion=True`

因此当前仓库不能输出正式 Casimir torque 结论。local-response distance scan 只作为
zero-torque baseline 和数值基准；finite-q 诊断也只属于 response 层公式与接口排查。

## 当前阅读入口

建议先读：

1. `docs/reports/current_project_status.md`
2. `docs/reports/finite_q_response_status.md`
3. `docs/reports/local_response_baseline_status.md`
4. `docs/notes/numerical_stability_summary.md`

当前最新 finite-q 结果：

- `validation/outputs/response/finite_q_raw_q0_consistency/`

历史结果入口：

- `validation/outputs/archive/`
