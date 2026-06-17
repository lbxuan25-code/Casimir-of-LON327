# Stage 5.12 small real-material energy prototype

## 为什么 5.12 可以第一次做真实材料能量积分原型

Stage 5.11c 已经生成并通过了一小批真实 LNO327 reflection grid points。每个点都包含

\[
R^{TE/TM}(i\xi_n,Q,\varphi)
\]

以及 pointwise trace-log hook-in 所需的信息。因此 Stage 5.12 可以第一次不重跑 response，直接读取这些离散 reflection matrices，做一个小规模真实材料能量积分原型。

## 为什么它仍然不是物理结果

本阶段只使用

\[
n=\{1,2,4\},\qquad Q=\{0.05,0.10,0.20\}\ {\rm nm}^{-1},
\qquad \varphi=\{0^\circ,45^\circ,90^\circ,135^\circ\}.
\]

这个网格非常稀疏，不包含 \(n=0\)，也没有 Matsubara 截断、\(Q_{\max}\)、角度网格或径向网格的收敛审计。因此输出只能叫

\[
F_{\rm proto}/A
\]

不能解释为 production Casimir energy。

## 原型积分做了什么

对每个输入点，读取已存储的 \(R^{TE/TM}\)，构造 identical-sheet prototype：

\[
M=I-e^{-2\kappa d}RR,
\]

并计算

\[
\log\det M.
\]

随后使用 sparse scaffold weights 做离散求和：

\[
\frac{F_{\rm proto}}{A}
=
k_BT
\sum_n
\sum_{Q,\varphi}
W_{Q,\varphi}
\log\det M.
\]

这里的 Matsubara 权重只是 prototype placeholder；因为没有 \(n=0\)，它不是完整有限温自由能。

## 为什么只能看流程是否跑通

Stage 5.12 的意义是确认：

- Stage 5.11c reflection grid 可以被可靠读取；
- 复数矩阵反序列化正确；
- pointwise trace-log helper 可以复用；
- 稀疏 \(Q,\varphi\) 权重和 partial contribution 输出流程可运行；
- separation scan 的数值没有 NaN/Inf，虚部保持很小。

它不能用于比较相、拟合参数、预测 torque 或报告真实能量。

## 下一步

下一步应做 material-grid convergence planning：

- 加入 \(n=0\) zero-mode audit；
- 扩展 Matsubara 网格并做 \(n_{\max}\) 收敛；
- 扩展 \(Q\) 网格并做 \(Q_{\max}\)、\(n_Q\) 收敛；
- 扩展角度网格并做 \(n_\varphi\) 收敛；
- 设计真实材料 \(R^{TE/TM}(i\xi_n,Q,\varphi)\) 或 \(\tilde\sigma(i\xi_n,Q,\varphi)\) 的 production grid。
