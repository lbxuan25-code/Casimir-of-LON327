# Stage 5.5b reflection-input tensor formatter

## 本阶段做什么

Stage 5.5b 读取 Stage 5.4b 输出的 dimensionless sheet conductivity

\[
\tilde\sigma_{ij}=Z_0\sigma^{SI,sheet}_{ij}
\]

并把它整理为每个 \((i\xi,\mathbf q)\) 点上的 reflection-input tensor。这里的输出是 tangential electric field basis 下的 \(R_E^{LT}\)，不是 Lifshitz trace-log、不是 Casimir energy、不是 force，也不是 torque。

## 从 model q 到 SI 波矢

Stage 5.4b 的 \(q_x,q_y\) 是 model momentum。使用 thin-film lattice convention：

\[
Q_x=\frac{q_x^{model}}{a_x},\qquad Q_y=\frac{q_y^{model}}{a_y},
\]

\[
Q=\sqrt{Q_x^2+Q_y^2}.
\]

当前默认 \(a_x=a_y=3.754\ \text{\AA}\)。

## 从 Matsubara energy 到 SI 虚频

输入中的 \(\Omega_{\rm eV}=\hbar\xi\)，因此

\[
\xi=\frac{\Omega_{\rm eV}E_{\rm eV}}{\hbar}.
\]

真空衰减常数为

\[
\kappa=\sqrt{Q^2+\xi^2/c^2}.
\]

## 局部 L/T 基底

\((L,T)\) 是每个 \(\mathbf Q\) 点的局部基底，不是全局固定坐标：

\[
L\parallel\mathbf Q,\qquad T=\hat z\times L.
\]

旋转矩阵定义为

\[
R_Q=\begin{pmatrix}\hat Q_x&\hat Q_y\\-\hat Q_y&\hat Q_x\end{pmatrix},
\]

并使用

\[
\tilde\sigma_{LT}=R_Q\tilde\sigma_{xy}R_Q^T.
\]

## 真空 admittance 和 reflection-input matrix

在 tangential electric \(L/T\) basis 下：

\[
Y_0=\mathrm{diag}\left(\frac{\xi}{c\kappa},\frac{c\kappa}{\xi}\right).
\]

formatter 输出

\[
R_E^{LT}=-(2Y_0+\tilde\sigma_{LT})^{-1}\tilde\sigma_{LT}.
\]

它满足

\[
E^{ref}_{LT}=R_E^{LT}E^{inc}_{LT}.
\]

注意：这还不是文献 TE/TM amplitude convention，尤其不能未经声明把 \(R_{LL}\) 直接称作 \(r_{pp}\)。

## 为什么仍不能进入 Casimir

Stage 5.5b 只做 tensor formatting。它没有计算 Lifshitz trace-log，没有计算 Casimir energy/force/torque，也没有实现实际多层 reflection/Casimir solver。下一步应是 TE/TM adapter convention audit。
