# Stage 5.9 Casimir energy integration grid planning scaffold

## 为什么 5.8 之后仍不能直接做真实 Casimir energy

Stage 5.8 只验证了单点 trace-log integrand：

\[
\mathcal I=\log\det\left[I-e^{-2\kappa d}R_1R_2\right].
\]

正式 Casimir free energy density 需要完整的 Matsubara 求和和二维面内动量积分：

\[
\frac{\mathcal F}{A}
=
k_BT
\sum_{n=0}^{\infty}{}'
\int
\frac{d^2Q}{(2\pi)^2}
\mathcal I(i\xi_n,\mathbf Q,d,\theta).
\]

Stage 5.9 只规划这些变量、网格和数据需求，不做真实 production energy、force 或 torque。

## 正式计算需要哪些变量

正式计算至少需要：

- Matsubara index \(n\) 和虚频 \(\xi_n\)；
- 面内动量大小 \(Q\)；
- 面内角度 \(\varphi\)；
- 板间距 \(d\)；
- plate 2 相对 plate 1 的材料旋转角 \(\theta\)；
- 每个 \((i\xi_n,Q,\varphi,\theta)\) 点上的 \(R^{TE/TM}\)。

## Matsubara 频率

有限温 Matsubara 频率为

\[
\xi_n=\frac{2\pi n k_BT}{\hbar},\qquad n=0,1,\dots.
\]

prime sum 使用

\[
w_0=\frac12,\qquad w_{n>0}=1.
\]

## \(\Omega_{\rm eV}\) 与 \(\xi\) 的转换

已有 response pipeline 常用

\[
\Omega_{\rm eV}=\frac{\hbar\xi}{E_{\rm eV}}.
\]

反向转换为

\[
\xi=\frac{\Omega_{\rm eV}E_{\rm eV}}{\hbar}.
\]

Stage 5.9 scaffold 会检查 \(\xi\rightarrow\Omega_{\rm eV}\rightarrow\xi\) 的 round trip。

## 极坐标测度

二维动量积分使用

\[
d^2Q=Q\,dQ\,d\varphi,
\]

因此 measure 为

\[
\frac{Q\,dQ\,d\varphi}{(2\pi)^2}.
\]

当前实现的权重只是 simple trapezoid scaffold，不是 production quadrature。后续必须做更可靠的 quadrature 和 convergence audit。

## 为什么 \(Q=0\) 是特殊点

在 \(Q=0\) 处，面内方向未定义，因此 TE/TM 的面内方向也不唯一。正式角度网格中需要用对称性、极限处理，或者把 \(Q=0\) 从普通 angular grid 中排除后单独处理。

## 为什么现有 8 个点不是生产网格

Stage 5.6/5.8 的代表性 rows 只用于 convention 和 integrand-level sanity check。它们不是覆盖 \((n,Q,\varphi)\) 的 integration grid，不能用于正式 energy integration。

## 材料旋转角如何影响 response grid

plate 2 的材料晶轴相对 lab frame 旋转 \(\theta\) 时，需要在材料自己的 crystal frame 中查询响应：

\[
\mathbf Q^{crystal}_2=R(-\theta)\mathbf Q^{lab}.
\]

得到的 response 或 reflection matrix 最终必须重新表达回共同 lab-frame TE/TM basis，才能进入 \(R_1R_2\)。

## 后续正式计算的数据需求

正式计算需要完整网格或可靠插值：

\[
\tilde\sigma(i\xi_n,Q,\varphi)
\]

或

\[
R^{TE/TM}(i\xi_n,Q,\varphi).
\]

此外还需要 \(n_{\max}\)、\(Q_{\max}\)、\(n_Q\)、\(n_\varphi\) 的收敛审计，高频和大 \(Q\) 截断审计，以及距离扫描误差控制。

下一步建议先做 toy-model full integration convergence audit，而不是直接真实材料 production energy。
