# Commutator sign and quadrature audit

## 1. 任务边界

本文档只做 commutator sign check、direct contact sign check、$C_j$ vs $K_j$
comparison，以及 mesh convergence / quadrature audit。它不修改 bubble formula，不修改主
response path，不引入 fitted contact，不进入 conductivity / reflection / Casimir，也不声明
Ward closure。

固定对象为

$$
V_i=\frac{\delta H}{\delta A_i},\qquad
M_{ij}=\frac{\delta^2H}{\delta A_i\delta A_j},
$$

$$
J=(\rho,-V_x,-V_y),\qquad P=(\rho,V_x,V_y).
$$

## 2. 固定定义

left bubble Ward contraction 的 spatial source column 为

$$
R^{\mathrm{bubble}}_{L,j}
=
i\Omega\Pi^{\mathrm{bubble}}_{0j}
+
q_i\Pi^{\mathrm{bubble}}_{ij}.
$$

候选 commutator expectation 使用两种 q-routing：

$$
C_j^{(+q)}
=
\sum_k
\operatorname{Tr}
\left[
\left(f(H_-)-f(H_+)\right)V_j(k,q)
\right],
$$

$$
C_j^{(-q)}
=
\sum_k
\operatorname{Tr}
\left[
\left(f(H_-)-f(H_+)\right)V_j(k,-q)
\right].
$$

direct contact contraction 定义为

$$
K_j(q)
=
\sum_k
\operatorname{Tr}
\left[
f(H_0(k))
\left(q_xM_{xj}(k,q)+q_yM_{yj}(k,q)\right)
\right].
$$

当前 direct residual 应满足

$$
R^{\mathrm{direct}}_{L,j}=-K_j(q).
$$

## 3. 需要数值验证的关系

Stage 4.11 数值脚本逐项比较：

$$
R^{\mathrm{bubble}}_{L,j}\stackrel{?}{=}+C_j^{(+q)},\qquad
R^{\mathrm{bubble}}_{L,j}\stackrel{?}{=}-C_j^{(+q)},
$$

$$
R^{\mathrm{bubble}}_{L,j}\stackrel{?}{=}+C_j^{(-q)},\qquad
R^{\mathrm{bubble}}_{L,j}\stackrel{?}{=}-C_j^{(-q)},
$$

以及

$$
R^{\mathrm{direct}}_{L,j}\stackrel{?}{=}-K_j(q),
$$

$$
C_j^{(+q)}\stackrel{?}{=}K_j(q).
$$

这些比较只用于诊断 sign / q-routing / quadrature，不用于选择或修改主公式。

## 4. Mesh convergence / quadrature

对固定方向 $q=(0.02,0.013)$ 及 scale

$$
1,\ 0.5,\ 0.25,\ 0.125
$$

在 mesh sizes

$$
8,\ 12,\ 16,\ 24,\ 32
$$

上比较 $C_j^{(+q)}-K_j$、$C_j^{(-q)}-K_j$ 和 $R^{\mathrm{total}}_{L,j}$。若误差随
mesh size 增大下降，log-log slope 应为负。该检查用于区分 finite mesh quadrature 与
解析 convention mismatch。

## 5. 结论表

| 项目 | 数值状态 | 结论 |
|---|---|---|
| bubble sign vs $+C^{(+q)}$ | 由 Stage 4.11 输出给出 | MATCH/MISMATCH |
| bubble sign vs $-C^{(+q)}$ | 由 Stage 4.11 输出给出 | MATCH/MISMATCH |
| bubble sign vs $+C^{(-q)}$ | 由 Stage 4.11 输出给出 | MATCH/MISMATCH |
| bubble sign vs $-C^{(-q)}$ | 由 Stage 4.11 输出给出 | MATCH/MISMATCH |
| direct sign $R^{direct}=-K$ | 由 Stage 4.11 输出给出 | MATCH/MISMATCH |
| $C^{(+q)}$ vs $K$ at fixed mesh | 由 Stage 4.11 输出给出 | MATCH/MISMATCH/QUADRATURE_LIMITED |
| mesh convergence of $C-K$ | 由 Stage 4.11 输出给出 | CONVERGING/NOT_CONVERGING/INCONCLUSIVE |
| likely next issue | 由 Stage 4.11 固定规则给出 | SIGN / Q_ROUTING / QUADRATURE / DENSITY_CONVENTION / UNRESOLVED |

若 direct sign 通过、bubble contraction sign 稳定，但 $C-K$ 不随 mesh 收敛，则下一步应优先
审计 density operator q convention、source reverse matrix element routing 和 scalar source
normalization，而不是加入 fitted $E^{ET}$。
