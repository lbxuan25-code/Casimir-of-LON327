# Stage 5.4a 电导单位转换说明

## 为什么需要解析单位链

Stage 5.1b 固定的是 model-level bilayer sheet conductivity：

\[
\sigma^{model}_{ij}(i\Omega)=-\frac{\Pi_{ij}(i\Omega)}{\Omega_{\rm eV}}.
\]

这个量还不是 SI sheet conductivity，也不是 reflection 边界条件会直接使用的 dimensionless sheet admittance。因此 Stage 5.4a 单独把

\[
\sigma^{model}_{ij}\rightarrow\sigma^{SI,sheet}_{ij}\rightarrow\tilde\sigma_{ij}
\]

代码化，并用 synthetic checks 验证。

## Peierls 矢势单位

Peierls phase 中的 model vector potential 与 SI vector potential 的关系是

\[
A_i^{model}=\frac{e a_i}{\hbar}A_i^{SI}.
\]

这里 \(a_i\) 是 \(i=x,y\) 方向 lattice length。由于 current response 对 vector potential 的归一化随方向带入 \(a_i\)，sheet conductivity 的 SI 缩放包含

\[
\frac{a_i a_j}{A_{\rm cell}}.
\]

## SI sheet conductivity

bilayer-normalized 2D sheet conductivity 的单位转换为

\[
\sigma^{SI,bilayer\ sheet}_{ij}
=\frac{e^2}{\hbar}
\frac{a_i a_j}{A_{\rm cell}}
\sigma^{model}_{ij}.
\]

对正交矩形晶格 \(A_{\rm cell}=a_xa_y\)。若 \(a_x=2a,a_y=a,A_{\rm cell}=2a^2\)，geometry tensor 是

\[
\begin{pmatrix}
2 & 1\\
1 & 1/2
\end{pmatrix}.
\]

因此矩形晶格下 \(xx\) 与 \(yy\) 不会都自动抵消；offdiag 的 \(a_xa_y/A_{\rm cell}\) 才等于 1。对正方晶格 \(a_x=a_y=a\)，所有分量都简化为

\[
\sigma^{SI,sheet}_{ij}=\frac{e^2}{\hbar}\sigma^{model}_{ij}.
\]

## dimensionless sheet conductivity

定义

\[
\tilde\sigma_{ij}=Z_0\sigma^{SI,sheet}_{ij}.
\]

这里 \(\tilde\sigma\) 是 dimensionless sheet conductivity / dimensionless sheet admittance，不是新的材料模型参数。后续统一使用 `sigma_tilde`，不要再用 \(g\) 作为符号。

正方晶格下

\[
\tilde\sigma_{ij}=Z_0\frac{e^2}{\hbar}\sigma^{model}_{ij}
=4\pi\alpha\,\sigma^{model}_{ij}.
\]

## 为什么还不能进入 reflection/Casimir

Stage 5.4a 只验证单位链。当前默认薄膜工作晶格常数来自统一结构配置：

\[
a_x=a_y=3.754\times10^{-10}\ {\rm m}=3.754\ \text{\AA},
\]

\[
A_{\rm cell}=1.4092516\times10^{-19}\ {\rm m^2}.
\]

这不是旧的 \(3.85\ \text{\AA}\) placeholder。若未来有样品特定晶格常数，应通过 material structure config 覆盖。Stage 5.4a 没有转换实际 Stage 5.2/5.3b scan 数据，也没有构造 reflection matrix、处理 \(n=0\) policy 或 finite-thickness slab。因此通过本阶段后也只是进入 Stage 5.4b 数据转换准备，仍不能声明 Casimir-ready。
