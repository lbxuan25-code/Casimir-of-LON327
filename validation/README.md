# Validation 指南

`validation/` 保存当前可复现的数值检验、诊断结果、轻量 status 和复现入口。这里不是 raw numerical artifacts 的长期仓库。

## 目录组织

- `scripts/`：可复现的 validation、diagnostic、convergence 和 smoke 检验入口；
- `outputs/`：按具体检验对象组织的 README、summary、status marker 和 command；
- `cache/`：可再生成的响应张量或中间数组缓存；
- `reports/`：跨主题 validation 总览和 artifact policy。

validation 报告按“具体检验对象”阅读：每个检验应说明检验目的、被检验对象、判据、结果、边界和复现方式。

BdG finite-q validation 关注 raw finite-q response 是否满足 Ward / gauge closure，以及它是否可进入 formal conductivity / reflection / Casimir gating chain。对应复现入口位于 `validation/scripts/bdg_finite_q/`，统一轻量报告和复现命令位于 `validation/outputs/finite_q_ward/`。

## Artifact 策略

长期进入 Git 的证据应保持紧凑：

- README；
- summary markdown；
- 小型 status / summary JSON 或 CSV；
- `command.sh` 或复现命令；
- validation report 文档。

默认不跟踪：

- `.npz` / `.npy`；
- raw、expanded 或大型 data CSV；
- cache tensors；
- intermediate outputs；
- repeated benchmark figures；
- scratch logs。

`validation/cache/` 总是可再生成。`validation/outputs/` 只保留小型摘要、状态和复现入口。需要复查 raw artifact 时，运行对应目录的 `command.sh` 或 `validation/scripts/**` 重新生成本地 ignored 输出。

## 阅读顺序

1. `validation/reports/validation_summary.md`
2. `validation/reports/validation_artifact_inventory.md`
3. `validation/outputs/**/README.md`
4. `validation/outputs/**/*summary*.md`
5. `validation/outputs/**/command.sh`

## 边界

validation 报告只说明当前检验结果和适用范围。除非 summary/status 明确写成 production-ready，否则 diagnostic-only 结果不得作为正式 Casimir energy、force、torque 输入。
