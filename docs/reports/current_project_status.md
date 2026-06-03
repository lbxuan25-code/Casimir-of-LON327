# 当前项目总状态

## 总览

local-response 数值稳定性阶段已经完成归纳。normal response 的 FS-adaptive sampling、
BdG response 基础稳定性、local-response Casimir integral convergence、refined cutoff /
Matsubara scan 以及 zero-torque baseline 均已形成 benchmark 级别证据。

local-response distance scan benchmark 已完成。normal、spm、dwave 三个 kind 在当前
local-response 设置下均保持 zero-torque baseline。

当前主线只保留 local q=0 response：normal Kubo、local BdG response、单位转换、
n=0 policy 和 local-response Casimir benchmark。有限动量 response prototype 已从
当前分支移除，后续如需重启必须重新设计闭合的 response 层。

本轮仓库结构已经完成 docs / scripts / outputs 分层整理。当前 `docs/reports/` 保留阶段报告，
`outputs/` 只保留 active 数据产物、cache 和 archive；旧 response、normal sampling、
smoke 和中间 Casimir convergence 输出已移动到 `validation/outputs/archive/`，移动清单见
`validation/outputs/archive/ARCHIVE_INDEX.md`。

## 当前边界

当前仓库不能输出正式 Casimir torque 结论。local-response distance scan 只作为
zero-torque baseline 和数值基准；`n0_policy=skip` 仍是重要边界。

## 当前阅读入口

建议先读：

1. `docs/reports/current_project_status.md`
2. `docs/reports/local_response_baseline_status.md`
3. `docs/notes/numerical_stability_summary.md`

当前最新 local-response 结果：

- `validation/outputs/casimir/local_response_integral/distance_scan/`

历史结果入口：

- `validation/outputs/archive/`
