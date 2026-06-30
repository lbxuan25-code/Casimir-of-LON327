# 脚本入口说明

`scripts/` 保存当前主计算入口。

- `normal_state/`：normal-state inspection 与 conductivity 计算；
- `pairing/`：pairing / gap structure 诊断；
- `bdg/`：BdG kernel 与 superconducting response 诊断；
- `casimir/`：基于当前 response contract 的 Casimir benchmark / baseline 计算。

`validation/scripts/` 保存收敛性、数值稳定性、static / `n=0` policy、response diagnostic、Casimir benchmark-only 和 smoke 检验入口。

## 新脚本放置规则

- 主计算入口放在 `scripts/`；
- 可信度、收敛性、diagnostic-only 和 benchmark-only 检验放在 `validation/scripts/`；
- 不在 `scripts/` 顶层新增 compatibility wrapper。

## 原则

- `scripts/` 面向当前主计算对象；
- `validation/scripts/` 面向可信度证据和复现检验；
- 脚本说明应写清楚输入、输出、适用边界和是否可作为 production input。
