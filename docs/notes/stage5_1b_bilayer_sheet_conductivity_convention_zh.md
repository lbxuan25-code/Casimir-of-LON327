# Stage 5.1b bilayer sheet conductivity convention

## 目的

Stage 5.1 的结论是 `CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE` 和 `UNIT_CHAIN_AMBIGUOUS`。本阶段不重新修改 response kernel，而是显式固定一个轻量的

\[
\Pi_{ij}(i\Omega)\rightarrow\sigma^{\rm model}_{ij}(i\Omega)
\]

约定，并把它限定为 bilayer-normalized 2D sheet conductivity。

## 2.1 为什么 response 先于 conductivity

底层 gauge response 是

\[
\Pi_{\mu\nu}=\frac{\delta\langle J_\mu\rangle}{\delta a_\nu},
\qquad
a_\nu=(\phi,A_x,A_y).
\]

它描述的是 observable current/density 对外源 gauge field 的线性响应。conductivity 不是新的 bubble，而是从 spatial block

\[
\Pi_{ij}=\frac{\delta\langle j_i\rangle}{\delta A_j}
\]

派生出来的电磁输出量。因此先要固定 \(\Pi_{\mu\nu}\) 的 source/observable convention、Ward identity 和 contact convention，然后才谈
\(\Pi_{ij}\rightarrow\sigma_{ij}\) 的符号与归一化。

本阶段保持既有约定不变：

\[
J=(\rho,-V_x,-V_y),\qquad P=(\rho,V_x,V_y),
\]

并保持 direct contact

\[
D_{ij}=-\langle M_{ij}\rangle.
\]

## 2.2 实时间到虚频推导

电场与势的关系为

\[
\mathbf E=-\nabla\phi-\partial_t\mathbf A.
\]

取 Fourier convention

\[
f(t)\sim e^{-i\omega t}.
\]

因此

\[
E_j(\omega,\mathbf q)=i\omega A_j(\omega,\mathbf q)-iq_j\phi(\omega,\mathbf q).
\]

在 transverse / optical gauge 或 \(\phi=0\) 时，

\[
E_j(\omega)=i\omega A_j(\omega).
\]

response 定义为

\[
j_i(\omega)=\Pi^R_{ij}(\omega)A_j(\omega).
\]

conductivity 定义为

\[
j_i(\omega)=\sigma^R_{ij}(\omega)E_j(\omega).
\]

因此

\[
\Pi^R_{ij}(\omega)=i\omega\sigma^R_{ij}(\omega),
\]

\[
\sigma^R_{ij}(\omega)=\frac{\Pi^R_{ij}(\omega)}{i\omega}.
\]

令

\[
\omega=i\xi,
\]

得到

\[
i\omega=-\xi.
\]

所以

\[
\boxed{
\sigma_{ij}(i\xi)=-\frac{\Pi_{ij}(i\xi)}{\xi}
}
\]

代码中使用 Matsubara energy variable

\[
\Omega_{\rm eV}=\hbar\xi,
\]

所以 model-level conversion 写作

\[
\boxed{
\sigma^{\rm model}_{ij}(i\Omega)
=-\frac{\Pi_{ij}(i\Omega)}{\Omega_{\rm eV}}
}
\]

该式只固定模型层的 \(\Pi\to\sigma\) 号和频率除法，不包含最终 SI sheet-conductivity scaling。

## 2.3 Bilayer-normalized 2D sheet conductivity

这里的 2D sheet conductivity 不等于 single-layer approximation。response \(\Pi_{ij}\) 仍然来自完整 bilayer Hamiltonian，因此层间 hopping、层间杂化、bonding/antibonding band，以及 bilayer pairing structure 都已经进入 \(\Pi_{ij}\)。

输出的

\[
\sigma^{\rm model}_{ij}(i\Omega)
\]

表示整个 bilayer 单元的面内 sheet response。它不是 3D bulk conductivity，也不是 single-layer conductivity。当前阶段也暂时不包含最终 SI scaling。

如果薄膜有 \(N\) 个等效 bilayer，后续阶段可以考虑

\[
\sigma^{\rm film}_{ij}=N\sigma^{\rm bilayer}_{ij}.
\]

如果要描述有限厚度、电磁场在层间的分布或 \(c\)-axis response，需要以后升级到 finite-thickness bilayer slab / multilayer transfer matrix；当前阶段不做。

## 2.4 当前阶段不能进入 Casimir

本阶段只固定 \(\Pi\to\sigma\) convention，不产生 reflection matrix，不产生 Casimir torque，也不判断 d-wave torque。

finite-q angular dependence 后续会保留为

\[
\sigma_{ij}(i\xi,\mathbf q),
\]

因此不能把所有 q-direction anisotropy 自动判定为错误。是否可进入 reflection / Casimir，需要在 Stage 5.2 之后继续检查 conductivity 数值行为、单位归一化和电磁边界条件。
