# 脚本入口说明

本目录只保留当前材料和模型本征特性的计算入口：

- `normal_state/`：normal-state inspection 与 conductivity 计算。
- `pairing/`：pairing / gap structure 诊断。
- `bdg/`：BdG kernel 与 superconducting response 诊断。
- `casimir/`：基于当前 local-response contract 的 Casimir 初级结论计算。

顶层不保留脚本副本或 compatibility wrapper。运行入口直接使用对应主题目录中的实现，
避免同一脚本出现多个路径。

## 已迁移到 validation

收敛性、数值稳定性、static / n=0 policy、Casimir convergence benchmark
和历史 smoke 入口都已移动到：

- `../validation/scripts/bdg_finite_q/`：当前 BdG finite-q validation workflows。
- `../validation/scripts/numerical_stability/`
- `../validation/scripts/response/`：非 BdG finite-q 的 response validation 与历史对照入口；当前 BdG finite-q workflow 不在这里。
- `../validation/scripts/casimir/`
- `../validation/scripts/smoke/`

已删除的历史 numbered workflow 不再作为 runnable entry 保留；Git history 是这些旧脚本的归档位置。

## 新脚本放置规则

- 材料本征结果脚本放入 `normal_state/`、`pairing/` 或 `bdg/`。
- 已形成初级结论的主计算脚本放入相应主题目录，例如 `casimir/`。
- 收敛性、诊断、benchmark-only 脚本放入 `../validation/scripts/`。
- 不在 `scripts/` 顶层新增脚本或 compatibility wrapper。

## 原则

- 不修改物理公式。
- `scripts/` 是当前材料计算入口；`validation/scripts/` 是可信度证据入口。
