# Stage 4.18 corrected Ward residual convention

## 目的

Stage 4.18 将 Stage 4.17 确认的 right Ward source-side residual 约定固化到 diagnostic helper、验证脚本和测试中，并用 Stage 4.16 的 adaptive full-response quadrature 重新验证完整 normal-state response。

本阶段只修正 Ward residual 的诊断定义，不修改主 response formula。

## 边界

- 不修改主 physical response formula。
- 不修改 Stage 4.13 后的 bubble prefactor sign。
- 不修改 Peierls vector/contact vertex。
- 不修改 source/observable split。
- 不修改 direct contact，仍使用 \(D_{ij}=-\langle M_{ij}\rangle\)。
- 不新增 fitted contact。
- 不新增 \(E^{ET}\)。
- 不进入 conductivity、reflection 或 Casimir。

## 修正后的 residual 约定

当前 physical response 使用

\[
J=(\rho,-V_x,-V_y),\qquad P=(\rho,V_x,V_y).
\]

left contraction 作用在 observable 侧。由于 observable spatial vertex 是 \(J_i=j_i=-V_i\)，历史 diagnostic 中的 left residual 保持

\[
R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu}.
\]

right contraction 作用在 source 侧。由于 source spatial vertex 是 \(P_i=V_i\)，并且

\[
G_+^{-1}-G_-^{-1}=i\Omega\rho-q_iV_i,
\]

right residual 应写成

\[
R_R[\mu]=i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.
\]

旧的

\[
R_R^{legacy}[\mu]=i\Omega\Pi_{\mu0}+q_x\Pi_{\mu x}+q_y\Pi_{\mu y}
\]

只保留为 old/legacy diagnostic comparison，不作为 closure 判据。

## 与前序阶段的关系

Stage 4.13 修正的是 bubble sign。Stage 4.15 解决的是 \(C-K\) 诊断中的低温 Fermi-window quadrature 问题。Stage 4.17 发现的是 right Ward diagnostic convention 问题。

Stage 4.18 不改变 response 本身，只将正确的 left/right residual convention 固化，并重新输出 full-response Ward validation。

## 后续

若 corrected left/right residual 在 adaptive full-response validation 中同时降到数值阈值以下，则说明之前的 right residual 主要是 diagnostic convention 问题。进入 conductivity、reflection 或 Casimir 前，仍需要 Stage 4.19 多参数 robustness scan。

