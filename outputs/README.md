# 输出指南

本目录只保留材料、模型本征特性和已形成初级结论的轻量说明文件，方便直接阅读当前物理对象。
raw 数组、scratch 图、cache、临时结果和大型 CSV 不在 Git 中跟踪；需要时由对应脚本重新生成。
数值收敛性、公式诊断、Casimir convergence benchmark 和 cache 已统一移动到 `../validation/`。

## 当前保留内容

- `normal_state/`：normal-state conductivity 与 band / block inspection 的 README 和可再生成目录。
- `pairing/`：pairing 和 gap structure 的 README 和可再生成目录。
- `bdg/`：BdG paramagnetic / diamagnetic / total kernel 与 superconducting response 的 README 和可再生成目录。
- `casimir/local_response_distance_scan/`：local-response baseline 的轻量说明与复现命令；仍跳过 n=0 且不含 finite-momentum response。

## 阅读顺序

1. `../docs/current_route.md`
2. `normal_state/README.md`
3. `pairing/gap_structure/README.md`
4. `bdg/total_kernel_imag/README.md`
5. `bdg/superconducting_response_imag/README.md`
6. `casimir/local_response_distance_scan/README.md`

## 验证材料位置

- 收敛性和数值稳定性：`../validation/outputs/numerical_stability/`
- response 诊断：`../validation/outputs/response/`
- 可复用中间张量 cache：`../validation/cache/`

## 维护原则

- `outputs/` 面向当前材料本征结果和边界清楚的初级结论。
- `validation/` 面向“为什么这些计算可信”的支撑证据。
- `outputs/**/data/` 和 `outputs/**/figures/` 只保留 `.gitkeep`；生成内容不提交。
- 需要追溯 validation 证据时先看 `../validation/README.md`。
