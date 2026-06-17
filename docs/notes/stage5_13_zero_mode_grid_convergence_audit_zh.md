# Stage 5.13 zero-mode and grid-convergence planning audit

## 本阶段做什么

Stage 5.13 在 Stage 5.12 小规模真实材料能量原型之后，专门审计正式 Casimir 计算前最容易出问题的两件事：

- \(Q\to0^+\) 极限；
- \(n=0\) zero-mode；
- 正式 \((n,Q,\varphi)\) 网格收敛规划。

本阶段不是 production Casimir energy calculation，不输出 force，也不输出 torque。

## 为什么 \(Q=0\) 不作为普通点

在 \(Q=0\) 处，面内方向未定义，因此 \(L/T\) 和 TE/TM 的面内方向也不唯一。正式积分应避免把 \(Q=0\) 当普通 angular-grid 节点，而应使用不含端点的内部 quadrature nodes，并通过 \(Q\to0^+\) 极限或对称性处理零点。

Stage 5.13 的 small-Q audit 使用 \(Q>0\) 的小值来检查 response、\(\tilde\sigma\)、\(R^{TE/TM}\) 和 \(\log\det M\) 是否有限且平滑。

## 为什么 \(n=0\) 不能直接代入

conductivity convention 使用

\[
\sigma^{model}_{ij}=-\Pi_{ij}/\Omega_{\rm eV}.
\]

因此不能直接把 \(\Omega=0\) 代入。zero-mode 应由 \(\xi\to0^+\) 的 \(R^{TE/TM}\) 极限得到，然后在 Matsubara prime sum 中使用 \(w_0=1/2\)。

Stage 5.13 使用一组小正 \(\Omega_{\rm eV}\) 来审计 zero-mode 极限趋势。

## 网格收敛规划

建议先分三档规划：

- coarse: \(n_{\max}=8, n_Q=16, n_\varphi=8\)
- medium: \(n_{\max}=16, n_Q=24, n_\varphi=12\)
- fine: \(n_{\max}=32, n_Q=32, n_\varphi=16\)

其中 \(Q_{\max}\)、\(n_{\max}\)、\(n_Q\)、\(n_\varphi\) 都需要单独收敛审计。正式材料 response grid 可以先用 direct response grid 做审计，再评估 interpolation grid 是否有足够误差控制。

## 边界

本阶段不修改 response formula、conductivity convention、reflection convention 或 trace-log convention。不输出 production energy、force 或 torque，也不声明 Casimir-ready。
