# Stage 5.4b SI sheet 与 sigma_tilde 数据转换

## Stage 5.4a 与 Stage 5.4b 的区别

Stage 5.4a 验证单位链 helper，本身只做 synthetic checks。Stage 5.4b 则读取已经验证过的 model conductivity JSON，把每个 \(\sigma^{model}_{ij}\) 转成 SI sheet conductivity 和 dimensionless sheet conductivity：

\[
\sigma^{SI,sheet}_{ij}
=\frac{e^2}{\hbar}\frac{a_i a_j}{A_{\rm cell}}\sigma^{model}_{ij},
\qquad
\tilde\sigma_{ij}=Z_0\sigma^{SI,sheet}_{ij}.
\]

## 为什么需要统一薄膜 lattice config

Stage 5.4a 之前的 \(3.85\ \text{\AA}\) 只是 placeholder。Stage 5.4b 使用统一 thin-film working config：

\[
a_x=a_y=3.754\ \text{\AA}=3.754\times10^{-10}\ {\rm m},
\]

\[
A_{\rm cell}=1.4092516\times10^{-19}\ {\rm m^2}.
\]

它描述的是 coherently strained thin-film LNO327 / LNO327-like film 的面内工作值，不是 relaxed bulk La3Ni2O7。若未来有样品特定晶格常数，应通过 material config override。

## 正方近似下的几何 tensor

当前 \(a_x=a_y\)，因此

\[
\frac{a_i a_j}{A_{\rm cell}}=1
\]

对 \(xx,xy,yx,yy\) 全部分量成立。数值上

\[
\sigma^{SI,sheet}_{ij}=\frac{e^2}{\hbar}\sigma^{model}_{ij},
\qquad
\tilde\sigma_{ij}=4\pi\alpha\,\sigma^{model}_{ij}.
\]

代码仍保留一般几何因子，以便以后处理 \(a_x\neq a_y\) 或非正方薄膜。

## 三种电导的区别

\(\sigma^{model}\) 是 Stage 5.1b 约定下的 bilayer-normalized model sheet conductivity。

\(\sigma^{SI,sheet}\) 是 one-bilayer sheet response，单位 Siemens。它不是 bulk 3D conductivity，也不是 single-layer conductivity。

\(\tilde\sigma\) 是 dimensionless sheet conductivity / dimensionless sheet admittance，定义为

\[
\tilde\sigma_{ij}=Z_0\sigma^{SI,sheet}_{ij}.
\]

后续统一使用 `sigma_tilde`，不再使用 \(g\) 作为符号。

## 为什么仍不是 reflection/Casimir

Stage 5.4b 只做数据单位转换，不构造 reflection matrix，不处理 \(n=0\) policy，不处理 finite-thickness slab，也不运行 response scan。因此它只是 reflection-input preparation 之前的数据准备阶段，仍不能声明 Casimir-ready。
