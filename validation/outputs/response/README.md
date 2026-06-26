# Response validation 轻量证据目录

本目录保存 response validation 的长期证据：中文摘要、状态 marker 和复现入口。它不是 raw numerical artifacts 仓库。

`data/`、`figures/`、`raw/`、`intermediate/` 默认视为可再生成输出，并由 `.gitignore` 排除。若未来必须保留大型 artifact，需要在 `validation/reports/validation_artifact_inventory.md` 或对应子目录 README 中说明原因。

## 目录组织

当前 response validation 按计算流程收敛为四个目录：

1. `normal_finite_q_kernel_convergence/`：normal finite-q current-current kernel 的同接口与收敛诊断。
2. `bdg_finite_q/`：finite-q BdG superconducting response 的 diagnostic 状态、失败边界和 Casimir input gate。
3. `ward_convention/`：Peierls vertex、contact term、density-current response convention 与 Ward residual convention 的约定审计。
4. `unit_reflection/`：response-to-sheet-conductivity、单位转换、reflection input、TE/TM adapter、q-grid / model-q mapping 的候选流程审计。

## 当前支撑证据

已可作为当前代码路径支撑证据的内容包括：

- normal-state finite-q kernel 的 q=0/q!=0 同接口一致性诊断；
- response / Ward convention 的 normal-state 约定审计；
- sheet conductivity、SI / dimensionless sheet scaling、reflection tensor formatting 和 TE/TM adapter 的格式与单位链候选检查；
- BdG finite-q engine 可用性、PairingAnsatz input layer 分离、normal limit 与 q->0 limit 诊断。

## 明确边界

以下内容仍是 diagnostic-only：

- raw finite-q BdG response；
- response-level / LSQ / quick audit 产生的修复式候选；
- 未经 Ward validation、unit policy、n=0 policy 同时闭合的 reflection input；
- 任何 stage5 prototype / scaffold / toy integration 输出。

这些结果不能直接作为正式 Casimir energy、force 或 torque input。当前 production 路径仍必须通过 consolidated validation marker 检查；marker 为 `FAILED` 时默认拒绝使用 finite-q BdG response，除非显式启用 diagnostic override。
