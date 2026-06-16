# Stage 5.8 Casimir trace-log integrand prototype

## integrand prototype 是什么

Stage 5.8 只在 synthetic toy matrices 和少量 Stage 5.6 validation-point reflection matrices 上计算单点

\[
\mathcal I(\xi,\mathbf Q,d)=
\log\det\left[I-e^{-2\kappa d}R_1^{TE/TM}R_2^{TE/TM}\right].
\]

这里

\[
\kappa=\sqrt{Q^2+\xi^2/c^2},
\qquad
e^{-2\kappa d}
\]

是 round-trip propagation factor。

本阶段的输出是 prototype integrand value，不是 Casimir energy、force 或 torque。

## 和正式 Casimir energy calculation 的区别

正式能量计算需要类似

\[
k_BT\sum_n'\int\frac{d^2Q}{(2\pi)^2}\mathcal I(i\xi_n,\mathbf Q,d,\theta)
\]

的完整 Matsubara 求和和二维 \(\mathbf Q\) 积分。Stage 5.8 不做这两件事，也不乘 \(k_BT\)、\(\hbar/2\pi\) 或 \(d^2Q/(2\pi)^2\) 的权重。

因此 Stage 5.8 只能回答：单点 trace-log integrand 的矩阵约定、符号、极限行为和 toy 角度依赖是否自洽。

## 为什么现在只能做 prototype

当前 validation pipeline 只有少量代表性点，主要用于 convention audit：

- Stage 5.6 给出少量 \(R^{TE/TM}\) rows；
- 这些 rows 不构成 \((\xi,Q,\varphi)\) 积分网格；
- 没有真实材料角度插值；
- 没有 Matsubara convergence；
- 没有 \(Q\)-grid convergence；
- 没有距离扫描和误差预算。

所以 8 个 validation points 不足以做正式 Casimir 积分。

## \(\mathcal M=I-e^{-2\kappa d}R_1R_2\) 的意义

\(\mathcal M\) 是两个反射矩阵和一次 round trip propagation 组成的 trace-log matrix。本阶段冻结：

\[
\mathcal M=I-e^{-2\kappa d}R_1R_2.
\]

其中 \(R_1\) 和 \(R_2\) 必须在同一个 lab-frame TE/TM amplitude basis 中表达。矩阵顺序使用 \(R_1R_2\)，不能只靠 determinant 检查，因为同维方阵有 \(\det(I-AB)=\det(I-BA)\)，但矩阵本身不同。

## synthetic checks 的物理含义

Stage 5.8 检查：

- zero reflection 给出 \(\mathcal I=0\)；
- one zero plate 给出 \(\mathcal I=0\)；
- large separation 时 \(\mathcal I\to0\)；
- weak scalar toy pair 在更小距离有更大的 \(|\mathcal I|\)；
- isotropic diagonal toy pair 满足解析公式；
- isotropic toy matrix 无角度依赖；
- anisotropic symmetric toy matrix 满足 \(\pi\) 周期性；
- 非对易 toy matrices 明确检查 \(R_1R_2\) 顺序。

其中 anisotropic toy periodicity 只用于 synthetic matrix sanity check，不是 LNO327 的物理 torque。

## 何时才可以进入正式能量计算

需要先规划并实现 material response grid：

\[
\tilde\sigma(i\xi_n,Q,\varphi)
\]

包括 Matsubara 网格、\(Q\) 网格、角度网格、材料旋转插值、距离依赖、收敛标准和误差控制。只有这些完成后，才适合进入正式 energy 或 torque calculation。

Stage 5.8 的下一步是 material response grid planning，而不是 production torque。
