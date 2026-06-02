# Outputs Guide

本目录只保留材料和模型本征特性的关键计算结果，方便直接阅读当前物理对象。
数值收敛性、公式诊断、Casimir benchmark、历史归档和 cache 已统一移动到
`../validation/`。

## 当前保留内容

- `normal_state/`：normal-state conductivity 与 band / block inspection 输出。
- `pairing/`：pairing 和 gap structure 输出。
- `bdg/`：BdG paramagnetic / diamagnetic / total kernel 与 superconducting response 输出。

## 阅读顺序

1. `../docs/reports/current_project_status.md`
2. `normal_state/README.md`
3. `pairing/gap_structure/README.md`
4. `bdg/total_kernel_imag/README.md`
5. `bdg/superconducting_response_imag/README.md`

## 验证材料位置

- 收敛性和数值稳定性：`../validation/outputs/archive/response/`,
  `../validation/outputs/archive/normal_state/`
- finite-q 诊断：`../validation/outputs/response/`,
  `../validation/outputs/archive/response/finite_q_*`
- Casimir benchmark：`../validation/outputs/casimir/`,
  `../validation/outputs/archive/casimir/`
- 可复用中间张量 cache：`../validation/cache/`

## 维护原则

- `outputs/` 面向当前材料本征结果。
- `validation/` 面向“为什么这些计算可信”的支撑证据。
- 不删除历史数据；需要追溯旧结果时先看 `../validation/README.md`。
