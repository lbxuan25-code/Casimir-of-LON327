# Validation 指南

`validation/` 保存验证逻辑、轻量结论和复现入口。这里不是 raw numerical artifacts 的长期仓库。

## 目录组织

- `scripts/`：可复现的 validation、diagnostic、convergence 和 smoke 脚本。
- `outputs/`：按具体检验对象组织的 README、summary、status marker 和 command。
- `cache/`：可再生成的响应张量或中间数组缓存。
- `reports/`：跨主题总览、artifact policy 和维护说明。

报告按“检验对象”阅读，不按历史 stage 编号阅读。已删除的历史 workflow 脚本由 Git history 归档；历史检验不变量由当前测试和当前 validation workflow 保留，而不是通过继续保留旧 runnable script 实现。

BdG finite-q blocker 的项目级 validation workflow 位于 `validation/scripts/bdg_finite_q/`。core finite-q 计算仍位于 `src/lno327/finite_q_engine.py`，基础可复用 helper 可保留在 `validation/lib/finite_q_diagnostics.py`。对应轻量输出和复现命令位于 `validation/outputs/bdg_finite_q/`。

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
