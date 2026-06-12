# Stage 4.15 Fermi-window adaptive quadrature

## 1. 背景

Stage 4.13 已修正 finite-q Kubo bubble 的 fermion-loop sign。修正后主路径满足

$$
R_L^{\mathrm{bubble}}[j]\approx +C_j,\qquad
R_L^{\mathrm{direct}}[j]\approx -K_j,
$$

因此剩余 left Ward residual 的 bookkeeping 为

$$
R_L^{\mathrm{total}}[j]\approx C_j-K_j.
$$

Stage 4.14 显示 $H_0(k)$ hopping reconstruction 和二阶 Peierls identity 均通过，
且升高温度会显著改善 $C-K$，所以最可疑来源是 low-temperature Fermi-surface
quadrature。

## 2. 本阶段目标

本阶段只验证 Fermi-window adaptive quadrature 能否改善

$$
C_j-K_j.
$$

其中

$$
C_j=\int_{\mathrm{BZ}}d^2k\,
\operatorname{Tr}\left[(f(H_-)-f(H_+))V_j(k,q)\right],
$$

$$
K_j=\int_{\mathrm{BZ}}d^2k\,
\operatorname{Tr}\left[f(H(k))q_iM_{ij}(k,q)\right],
$$

且

$$
H_\pm=H(k\pm q/2).
$$

## 3. Adaptive rule

1. 先将 BZ 分成 coarse grid cells。
2. 对 cell 的角点和中心点检查 $H(k)$、$H(k+q/2)$、$H(k-q/2)$ 的所有 band energies。
3. 若任一点满足 $|E_n-\mu|<\mathrm{fermi\_window}$，则标记为 Fermi-window cell。
4. 只细分被标记 cell；非标记 cell 保持原尺寸。
5. 每个最终 cell 用 tensor-product Gauss-Legendre quadrature。
6. $C_j$ 和 $K_j$ 必须使用完全相同的积分点和权重。

## 4. 判读

若 adaptive refinement 明显降低 30 K 下的 $|C-K|$，则说明 Stage 4.14 的剩余
residual 主要是 Fermi-surface quadrature 问题。下一步才考虑把同类积分策略用于完整
response diagnostic。

若 adaptive quadrature 无法改善，则应继续审计 finite-q density vertex embedding、
contact expectation routing 或其它 response-level convention。

## 5. 边界

本阶段不修改主 response，不修改 bubble sign，不修改 direct contact，不做 residual
tuning，不加入 fitted contact，不加入 $E^{ET}$，不进入 conductivity / reflection /
Casimir，也不声明 Ward closure。
