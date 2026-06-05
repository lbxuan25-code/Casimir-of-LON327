# 当前项目总状态

## 总览

local-response 数值稳定性阶段已经完成归纳。normal response 的 FS-adaptive sampling、
BdG response 基础稳定性、local-response Casimir integral convergence、refined cutoff /
Matsubara scan 以及 zero-torque baseline 均已形成 benchmark 级别证据。

local-response distance scan 已形成初级结论。normal、spm、dwave 三个 kind 在当前
local-response 设置下均保持 zero-torque baseline。

当前主线仍以 local q=0 response 为 Casimir baseline：normal Kubo、local BdG
response、单位转换、n=0 policy 和 local-response Casimir 初级结论。finite-q
当前只保留 Stage 1 normal current-current kernel convergence diagnostic；它不是
gauge/Ward-closed finite-q conductivity，也不作为 Casimir 输入。

本轮仓库结构已经完成 docs / scripts / outputs 分层整理。当前 `docs/reports/`
保留阶段报告，`outputs/` 只保留 active 数据产物与初级结论；validation 诊断输出
位于 `validation/outputs/`，cache 位于 `validation/cache/`。旧 mixed sigma/K
diagnostics 已删除，不作为 validation evidence。

## 当前边界

当前仓库不能输出最终 Casimir torque 结论。local-response distance scan 是当前
zero-torque baseline 的初级结论；`n0_policy=skip` 与 finite-momentum response
未包含仍是重要边界。

## 当前阅读入口

建议先读：

1. `docs/notes/foundation_notes.md`
2. `docs/notes/numerical_stability_summary.md`
3. `docs/reports/local_response_baseline_status.md`
4. `docs/notes/finite_q_response_plan_zh.md`

## 当前 active outputs

- `outputs/casimir/local_response_distance_scan/`
- `validation/outputs/response/normal_finite_q_kernel_convergence/`
