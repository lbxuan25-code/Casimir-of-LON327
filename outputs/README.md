# 输出指南

`outputs/` 保存当前主计算产物的轻量说明、summary、figures/data 目录占位和必要复现入口。

raw 数组、scratch 图、cache、临时结果和大型 CSV 不作为 `outputs/` 的长期 Git 内容。数值收敛性、公式诊断、benchmark-only 证据和 cache 属于 `validation/` 或本地再生成产物。

## 当前保留内容

- `normal_state/`：normal-state conductivity 与 band / block inspection 的说明；
- `pairing/`：pairing 和 gap structure 的说明；
- `bdg/`：BdG paramagnetic / diamagnetic / total kernel 与 superconducting response 的说明；
- `casimir/local_response_distance_scan/`：local-response baseline / benchmark 的轻量说明与复现入口。

## 阅读顺序

1. `../docs/current_route.md`
2. `normal_state/README.md`
3. `pairing/gap_structure/README.md`
4. `bdg/total_kernel_imag/README.md`
5. `bdg/superconducting_response_imag/README.md`
6. `casimir/local_response_distance_scan/README.md`

## 维护原则

- `outputs/` 面向当前主计算产物和边界清楚的结果摘要；
- `validation/` 面向“为什么这些计算可信”的支撑证据；
- `outputs/**/data/` 和 `outputs/**/figures/` 只保留目录占位；生成内容不提交；
- 需要复查 validation 证据时先看 `../validation/README.md`。
