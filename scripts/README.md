# scripts 入口说明

本目录只保留当前材料和模型本征特性的计算入口：

- `normal_state/`：normal-state inspection 与 conductivity 计算。
- `pairing/`：pairing / gap structure 诊断。
- `bdg/`：BdG kernel 与 superconducting response 诊断。

根目录下的 `*.py` 文件是这些主题目录的 compatibility wrapper，用来保持短命令可用。

## 已迁移到 validation

收敛性、数值稳定性、finite-q 公式诊断、static / n=0 policy、Casimir benchmark
和历史 smoke 入口都已移动到：

- `../validation/scripts/numerical_stability/`
- `../validation/scripts/finite_q_diagnostics/`
- `../validation/scripts/response/`
- `../validation/scripts/casimir/`
- `../validation/scripts/smoke/`
- `../validation/scripts/compat/`

## 新脚本放置规则

- 材料本征结果脚本放入 `normal_state/`、`pairing/` 或 `bdg/`。
- 收敛性、诊断、benchmark-only 脚本放入 `../validation/scripts/`。
- 只有材料本征结果需要短命令时才在根目录新增 wrapper。
- wrapper 不放物理或绘图逻辑，只转发到主题目录实现。

## 原则

- 不删除历史脚本。
- 不修改物理公式。
- `scripts/` 是当前材料计算入口；`validation/scripts/` 是可信度证据入口。
