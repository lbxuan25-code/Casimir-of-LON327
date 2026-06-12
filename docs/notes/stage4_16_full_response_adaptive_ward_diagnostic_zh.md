# Stage 4.16 Full response adaptive Ward diagnostic

## 1. Boundary

本阶段只做 diagnostic。它不修改主 response，不修改 Stage 4.13 的 bubble sign，不修改
$V_i$、$M_{ij}$、$j_i=-V_i$，不修改 source/observable split，不修改 direct contact，
不加入 fitted contact，不加入 $E^{ET}$，不进入 conductivity / reflection / Casimir。

## 2. Formula being tested

Stage 4.13 已修正 bubble sign。当前 physical response 使用

$$
J=(\rho,-V_x,-V_y),\qquad P=(\rho,V_x,V_y),
$$

并计算

$$
\Pi_{\mu\nu}=\Pi_{\mu\nu}^{\mathrm{bubble}}+D_{\mu\nu}.
$$

bubble 使用 corrected positive fermion-loop prefactor；spatial-spatial direct contact 仍为

$$
D_{ij}=-\langle M_{ij}\rangle.
$$

本阶段检查

$$
R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu},
$$

$$
R_R[\mu]=i\Omega\Pi_{\mu0}+\Pi_{\mu x}q_x+\Pi_{\mu y}q_y.
$$

## 3. Adaptive quadrature summary

Stage 4.15 已证明 Fermi-window adaptive quadrature 显著改善 $C_j-K_j$。Stage 4.16
把同一套 adaptive points 和 weights 用于完整 $\Pi_{\mu\nu}$ 的 bubble 与 direct term，
检查 full Ward residual 是否随 refinement 系统下降。

## 4. Interpretation

如果 adaptive final 的 full Ward residual 明显小于 uniform mesh 64，说明低温
Fermi-surface quadrature 是主要误差来源。若 residual 仍未达到数值闭合阈值，则下一步应
继续在完整 response diagnostic 中提高 adaptive resolution 或审计 remaining routing /
density convention。

如果未闭合，不允许回退 Stage 4.13 bubble sign，也不允许改 direct contact。进入
conductivity / reflection / Casimir 前必须完成 full Ward response 数值验证。
