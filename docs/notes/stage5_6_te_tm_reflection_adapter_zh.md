# Stage 5.6 TE/TM reflection adapter

## 本阶段做什么

Stage 5.6 在 Stage 5.5b 的基础上，把内部 tangential electric basis 下的

\[
R_E^{LT}
\]

转换为 TE/TM amplitude basis 下的

\[
R^{TE/TM}.
\]

输入仍然来自 dimensionless sheet conductivity：

\[
\tilde\sigma_{ij}=Z_0\sigma^{SI,sheet}_{ij}.
\]

本阶段只做 adapter convention audit。它没有计算 Lifshitz trace-log，没有计算 Casimir energy、force、torque，也没有声明已经可以直接做完整 Casimir 计算。

## 内部 L/T 审计路径

对每个非零面内波矢 \(\mathbf Q=(Q_x,Q_y)\)，定义

\[
L\parallel \mathbf Q,\qquad T=\hat z\times L.
\]

若

\[
\hat e_L=(\hat Q_x,\hat Q_y),
\]

则

\[
\hat e_T=(-\hat Q_y,\hat Q_x).
\]

Stage 5.5b 使用的内部链路为

\[
\tilde\sigma_{xy}\rightarrow \tilde\sigma_{LT}\rightarrow R_E^{LT}.
\]

这里

\[
E_{LT}^{ref}=R_E^{LT}E_{LT}^{inc},
\]

且 \(R_E^{LT}\) 的行列顺序是 \((L,T)\)。

## TE/TM amplitude convention

Stage 5.6 输出给后续标准 trace-log convention 的矩阵使用 \((s,p)=(TE,TM)\) 顺序。

本文采用

\[
E_s^{inc}=E_T^{inc},\qquad E_s^{ref}=E_T^{ref},
\]

\[
E_p^{inc}=E_L^{inc},\qquad E_p^{ref}=-E_L^{ref}.
\]

其中 \(p\)-polarization 反射振幅的负号来自反射 TM 波传播方向反转后的振幅 convention。这是 metadata 和文档中显式记录的约定。

## Adapter 公式

设

\[
R_E^{LT}=
\begin{pmatrix}
R_{LL} & R_{LT}\\
R_{TL} & R_{TT}
\end{pmatrix}.
\]

则

\[
R^{TE/TM}=
\begin{pmatrix}
R_{TT} & R_{TL}\\
-R_{LT} & -R_{LL}
\end{pmatrix}.
\]

因此：

- \(R_{TT}\) 是 \(s\to s\)；
- \(R_{TL}\) 是 \(p\to s\)；
- \(-R_{LT}\) 是 \(s\to p\)；
- \(-R_{LL}\) 是 \(p\to p\)。

## Scalar limit

对 isotropic scalar sheet，

\[
R_{LL}^{E,LT}=-\frac{\tilde\sigma}{2\eta_L+\tilde\sigma},
\qquad
R_{TT}^{E,LT}=-\frac{\tilde\sigma}{2\eta_T+\tilde\sigma}.
\]

adapter 后得到

\[
r_{ss}=-\frac{\tilde\sigma}{2\eta_T+\tilde\sigma},
\qquad
r_{pp}=+\frac{\tilde\sigma}{2\eta_L+\tilde\sigma}.
\]

这个 \(r_{pp}\) 的正号是 Stage 5.6 最重要的 convention check 之一。

## Strong 和 weak sheet limit

当 \(\tilde\sigma\gg1\) 且远大于对应真空 admittance 时：

\[
r_{ss}\rightarrow -1,\qquad r_{pp}\rightarrow +1.
\]

当 \(\tilde\sigma\ll1\) 时：

\[
r_{ss}\approx-\frac{\tilde\sigma}{2\eta_T},
\qquad
r_{pp}\approx+\frac{\tilde\sigma}{2\eta_L}.
\]

validation script 会显式检查这两个极限。

## Offdiag mixing 和 Hall-like marker

symmetric offdiag tensor 会在 \(R_E^{LT}\) 和 \(R^{TE/TM}\) 中保留 offdiag mixing，并被标记为 symmetric finite-q mixing。

antisymmetric synthetic tensor 只作为 Hall-like marker 的符号检查使用。这个 marker 不被用于对真实 LNO327 数据作物理断言。

## 明确边界

Stage 5.6 没有修改 response formula、\(\Pi\rightarrow\sigma^{model}\) convention、bubble sign、direct contact、source/observable split、Ward residual convention。没有 fitted contact，没有新增 \(E^{ET}\)，没有运行 heavy response，没有计算 Lifshitz trace-log，也没有计算 Casimir energy、force、torque。
