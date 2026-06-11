# Equal-time / commutator Ward audit

## 1. 任务边界

本文档只审计 finite-q Ward closure 中 possible equal-time / commutator completion。
它不修改 bubble factor，不使用 residual tuning，不进入 conductivity / reflection /
Casimir，也不声明 Ward identity 已闭合。

固定对象为

$$
V_i=\frac{\delta H}{\delta A_i},\qquad
M_{ij}=\frac{\delta^2H}{\delta A_i\delta A_j},
$$

$$
J_0=\rho,\qquad J_i=-V_i,
$$

$$
P_0=\rho,\qquad P_i=V_i.
$$

response 定义为

$$
\Pi_{\mu\nu}
=
\frac{\delta\langle J_\mu\rangle}{\delta a_\nu}
=
-\langle J_\mu P_\nu\rangle_{\mathrm{bubble}}
+
\left\langle\frac{\delta J_\mu}{\delta a_\nu}\right\rangle
+
E_{\mu\nu}^{\mathrm{ET}}.
$$

本任务判断 $E_{\mu\nu}^{\mathrm{ET}}$ 是否能被证明为零；若不能证明，则标记为
`UNRESOLVED`，不得硬改主 response path。

## 2. 从 Ward contraction 推导

physical Ward contraction 使用

$$
Q_{\mathrm{phys}}=(i\Omega,q_x,q_y),
$$

$$
R_L[\nu]=Q_{\mathrm{phys},\mu}\Pi_{\mu\nu}.
$$

对 Stage 4.8 response，

$$
\Pi_{\mu\nu}^{4.8}
=
-\langle J_\mu P_\nu\rangle
+
\left\langle\frac{\delta J_\mu}{\delta a_\nu}\right\rangle .
$$

在 imaginary-time formalism 中，对

$$
Q_{\mathrm{phys},\mu}\left[-\langle J_\mu P_\nu\rangle\right]
$$

做时间导数/continuity 操作时，$i\Omega$ 项可产生 equal-time boundary term。其候选形式
包含 density 与 source vertex 的等时对易子，例如

$$
E_{\nu}^{\mathrm{ET,left}}
\sim
\langle[\rho_{\mathbf q},P_\nu]\rangle,
$$

具体符号、momentum routing 和 density normalization 依赖仓库尚未完全固定的 scalar
source convention 与 finite-q density vertex convention。因此当前解析状态为：

```text
UNRESOLVED
```

这不是说一定存在额外项；而是说 Stage 4.8/4.9 尚未证明
$E_{\mu\nu}^{\mathrm{ET}}=0$。

## 3. direct derivative term 与 commutator term

direct derivative term 定义为

$$
D_{\mu\nu}
=
\left\langle\frac{\delta J_\mu}{\delta a_\nu}\right\rangle .
$$

它与 possible equal-time / commutator completion

$$
E_{\mu\nu}^{\mathrm{ET}}
$$

不是同一个对象。

已知 spatial-spatial direct derivative term 为

$$
D_{ij}=-\langle M_{ij}\rangle.
$$

当前代码已经包含 $D_{ij}$。本任务审计的是 bubble + direct 后是否仍需要
$E_{ij}^{\mathrm{ET}}$ 或其它 block 的 equal-time completion。

## 4. spatial source column

Stage 4.9 显示主 residual 是 left / spatial / longitudinal，因此重点对象是

$$
R_L[j]
=
i\Omega\Pi_{0j}+q_i\Pi_{ij}.
$$

拆分为

$$
R_L[j]
=
R_L^{\mathrm{bubble}}[j]
+
R_L^{\mathrm{direct}}[j]
+
R_L^{\mathrm{ET}}[j].
$$

其中当前代码已经包含

$$
R_L^{\mathrm{direct}}[j]
=
q_iD_{ij}
=
q_i(-\langle M_{ij}\rangle).
$$

Stage 4.10 的 diagnostic 输出定义

$$
R^{\mathrm{missing}}
=
-
\left(R^{\mathrm{bubble}}+R^{\mathrm{direct}}\right).
$$

若只缺一个 equal-time / commutator completion，则其 Ward contraction 必须贡献

$$
R^{\mathrm{ET}}=R^{\mathrm{missing}}.
$$

## 5. 二阶 Peierls Ward identity

Peierls 顶角为

$$
V_j(k,q)
=
i\sum_R R_j t_R e^{ik\cdot R}
\operatorname{sinc}\left(\frac{q\cdot R}{2}\right),
$$

$$
M_{ij}(k,q)
=
-\sum_R R_iR_jt_R e^{ik\cdot R}
\operatorname{sinc}^2\left(\frac{q\cdot R}{2}\right).
$$

由 hopping formula 直接得到二阶 identity：

$$
q_iM_{ij}(k,q)
=
V_j(k+q/2,q)-V_j(k-q/2,q).
$$

推导中 $q\cdot R=2x$，右侧为

$$
iR_jt_Re^{ik\cdot R}
\left(e^{ix}-e^{-ix}\right)\operatorname{sinc}(x)
=
-2R_jt_Re^{ik\cdot R}\frac{\sin^2 x}{x},
$$

左侧为

$$
-(q\cdot R)R_jt_Re^{ik\cdot R}\operatorname{sinc}^2(x)
=
-2R_jt_Re^{ik\cdot R}\frac{\sin^2 x}{x}.
$$

因此解析状态为 `MATCH`。Stage 4.10 数值脚本检查该 identity 的最大误差。

## 6. 结论表

| 项目 | 解析状态 | 代码/诊断状态 | 结论 |
|---|---|---|---|
| $V_i$ vertex-level Ward identity | 已由 Peierls 一阶 identity 推导 | Stage 4.1B / tests | MATCH |
| second-order Peierls identity $q_iM_{ij}=\Delta V_j$ | 本文档推导 | Stage 4.10 script | MATCH |
| direct derivative term $D_{ij}=-\langle M_{ij}\rangle$ | 已由 $J_i=-\delta H/\delta A_i$ 推导 | 当前主 response 包含 direct component | MATCH |
| residual after bubble + direct | 不能由 direct term 自动保证为零 | Stage 4.9 / 4.10 result | UNRESOLVED |
| need for $E^{ET}$ | 尚未证明 $E^{ET}=0$ | missing residual diagnostic | UNRESOLVED |

当前结论：finite-q contact vertex 与 direct derivative term 本身不是主要疑点；剩余
left / spatial / longitudinal $O(q)$ residual 需要 explicit equal-time commutator
推导，而不是修改 bubble signs 或拟合 contact 系数。
