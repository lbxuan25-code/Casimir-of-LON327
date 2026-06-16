# Stage 5.7 pre-Lifshitz readiness audit

## 为什么 5.6 后还不能直接算 torque

Stage 5.6 已经把内部 tangential electric basis 的 \(R_E^{LT}\) 转换为 TE/TM amplitude basis 的

\[
R^{TE/TM}=
\begin{pmatrix}
r_{ss} & r_{sp}\\
r_{ps} & r_{pp}
\end{pmatrix}.
\]

但这只固定了单个 \((i\xi,\mathbf Q)\) 点上的反射矩阵 convention。真正的 Casimir torque 还需要完整的 Matsubara sum、二维 \(\mathbf Q\) 积分、角度依赖材料响应网格、数值收敛和误差控制。Stage 5.7 只检查 trace-log integrand 的矩阵约定，不做 production physics run。

## \(R^{TE/TM}\) 如何进入 \(\log\det\)

两个平行 2D sheets 中间是真空间隙，固定虚频 \(\xi\)、面内波矢 \(\mathbf Q\) 和距离 \(d\) 时：

\[
\kappa=\sqrt{Q^2+\xi^2/c^2}.
\]

传播因子为

\[
u=e^{-\kappa d},
\]

round-trip 因子为

\[
u^2=e^{-2\kappa d}.
\]

本阶段冻结的 trace-log matrix convention 是

\[
\mathcal M=I-e^{-2\kappa d}R_1^{TE/TM}R_2^{TE/TM}.
\]

单点 integrand 为

\[
\mathcal I=\log\det\mathcal M.
\]

这里的 \(\mathcal I\) 只是 integrand-level object，不是 Casimir energy、force 或 torque。

## 为什么要冻结行列 convention

TE/TM 矩阵采用 \((s,p)=(TE,TM)\) 顺序：

\[
\begin{pmatrix}
E_s^{ref}\\
E_p^{ref}
\end{pmatrix}
=
R^{TE/TM}
\begin{pmatrix}
E_s^{inc}\\
E_p^{inc}
\end{pmatrix}.
\]

因此矩阵的行是 reflected polarization，列是 incident polarization。这个约定必须在进入 trace-log 前固定，否则 offdiag mixing 的物理含义和矩阵乘法顺序会变得不透明。

## 为什么要冻结 \(R_1R_2\) 的顺序

Stage 5.7 采用

\[
\mathcal M=I-u^2R_1R_2.
\]

其中 \(R_1\) 是 lower plate，\(R_2\) 是 upper plate，二者都表示从真空间隙一侧看向对应 sheet 的 reflection matrix，并且必须在同一个 lab-frame TE/TM basis 中表达。

虽然对同维方阵有 \(\det(I-AB)=\det(I-BA)\)，但矩阵 \(\mathcal M\) 本身不同。为了后续调试 polarimetric mixing 和多层结构，必须现在就冻结 `R1 @ R2` 的顺序。

## 材料旋转角 convention

设 lab-frame 面内波矢为

\[
\mathbf Q^{lab}=(Q_x,Q_y).
\]

plate 1 的晶格坐标与 lab frame 对齐：

\[
\theta_1=0.
\]

plate 2 相对 plate 1 旋转 \(\theta\)。在第 \(a\) 个材料自己的 crystal frame 中：

\[
\mathbf Q_a^{crystal}=R(-\theta_a)\mathbf Q^{lab}.
\]

也就是说，旋转材料等价于用反向旋转的 \(\mathbf Q\) 查询材料响应。最终所有 \(R^{TE/TM}\) 仍然必须重新表达回共同的 lab-frame TE/TM basis。

Stage 5.7 只做 toy/synthetic rotation checks，不做真实材料响应插值。

## 和 production Casimir calculation 的区别

Stage 5.7 不做：

- full Matsubara sum；
- full \(d^2Q\) integral；
- Casimir energy；
- Casimir force；
- Casimir torque；
- \(s_\pm\) 与 \(d\)-wave 比较；
- substrate 或 finite-thickness 多层模型。

后续真正计算需要真实 \((\xi,Q,\varphi)\) 网格、材料角度插值、距离扫描、误差控制和收敛审计，而不是只用当前 validation pipeline 中的 8 个 representative cases。
