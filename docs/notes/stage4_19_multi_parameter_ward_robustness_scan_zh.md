# Stage 4.19 multi-parameter Ward robustness scan

## 目的

Stage 4.19 对 corrected full-response Ward validation 做多参数稳健性扫描，检查 Stage 4.18 的数值闭合是否对温度、Matsubara index、\(\mathbf q\) 方向和尺度、adaptive refinement、Gauss order、Fermi window 保持稳定。

默认脚本运行使用 representative scan：围绕基准点逐轴覆盖所有默认参数取值，避免完整 Cartesian scan 在常规验证中变成小时级任务。完整 Cartesian scan 仍可用 `--cartesian-full` 显式触发。

本阶段仍是 diagnostic-only。通过本阶段只表示 normal-state response Ward validation 在扫描网格上稳健，不表示 conductivity、reflection 或 Casimir 已完成，也不表示结果已经 Casimir-ready。

## 边界

- 不修改主 physical response formula。
- 不修改 Stage 4.13 后的 bubble prefactor sign。
- 不修改 \(V_i\)、\(M_{ij}\)、\(j_i=-V_i\)。
- 不修改 source/observable split。
- 不修改 direct contact，仍使用 \(D_{ij}=-\langle M_{ij}\rangle\)。
- 不新增 fitted contact。
- 不新增 \(E^{ET}\)。
- 不进入 conductivity、reflection 或 Casimir。
- 不声明 Casimir-ready。

## corrected Ward residual

当前 diagnostic convention 为

\[
R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu},
\]

\[
R_R[\mu]=i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.
\]

右侧 \(-q\) 号来自 source-side vertex \(P_i=V_i\) 以及

\[
G_+^{-1}-G_-^{-1}=i\Omega\rho-q_iV_i.
\]

## 与前序阶段的关系

Stage 4.13 修正 bubble sign。Stage 4.15 解决 \(C-K\) 诊断中的 Fermi-window quadrature 问题。Stage 4.17/4.18 修正 right Ward diagnostic convention，并在基准参数下确认 corrected residual 闭合到 \(10^{-7}\) 量级。

Stage 4.19 的目标是确认这个闭合不是单一参数点的偶然结果。

## 判定

单个参数组合：

- `CLOSED`: \(\max(|R_L|,|R_R|)<10^{-6}\)
- `ACCEPTABLE_BUT_MONITOR`: \(10^{-6}\le \max(|R_L|,|R_R|)<10^{-5}\)
- `NOT_CLOSED`: \(\max(|R_L|,|R_R|)\ge10^{-5}\)

全局：

- `ROBUSTLY_CLOSED`: 所有组合均为 `CLOSED`
- `MOSTLY_CLOSED_WITH_MINOR_OUTLIERS`: 95% 以上组合为 `CLOSED`，其余为 `ACCEPTABLE_BUT_MONITOR`
- `ROBUSTNESS_FAILURE`: 存在 `NOT_CLOSED`

## 后续

若 Stage 4.19 通过，下一步才可以进入 response-to-conductivity 的独立验证阶段。该后续阶段仍需要独立验证 conductivity convention、单位、analytic continuation / Matsubara routing，以及是否可作为 reflection/Casimir input。
