# Stage 4.14 C_j versus K_j routing/contact audit

## 1. Current post-Stage-4.13 bookkeeping

Stage 4.13 已修正 finite-q Kubo bubble 的 fermion-loop overall sign。当前主路径的
left Ward spatial-source bookkeeping 为

$$
R_L^{\mathrm{bubble}}[j]\approx +C_j,
$$

$$
R_L^{\mathrm{direct}}[j]\approx -K_j,
$$

因此

$$
R_L^{\mathrm{total}}[j]\approx C_j-K_j.
$$

剩余 residual 不是 bubble sign 问题，而是 $C_j-K_j$ 问题。

## 2. Definitions

定义

$$
C_j(q)
=
\sum_k
\operatorname{Tr}
\left[
\left(f(H_-)-f(H_+)\right)V_j(k,q)
\right],
$$

其中

$$
H_\pm=H_0(k\pm q/2).
$$

定义

$$
K_j(q)
=
q_i\langle M_{ij}\rangle
=
\sum_k
\operatorname{Tr}
\left[
f(H(k))
\left(q_xM_{xj}(k,q)+q_yM_{yj}(k,q)\right)
\right].
$$

## 3. Expected analytic identity

二阶 Peierls identity 给出

$$
q_iM_{ij}(k,q)
=
V_j(k+q/2,q)-V_j(k-q/2,q).
$$

因此

$$
K_j
=
\sum_k
\operatorname{Tr}
\left[
f(H(k))
\left(
V_j(k+q/2,q)-V_j(k-q/2,q)
\right)
\right].
$$

在连续 BZ 积分中，如果可以平移变量，

$$
K_j
=
\sum_k
\operatorname{Tr}
\left[
\left(f(H_-)-f(H_+)\right)V_j(k,q)
\right]
=
C_j.
$$

## 4. Possible failure modes

可能导致 $C_j\ne K_j$ 的来源包括：

1. $H_0(k)$ 和 hopping/Fourier reconstruction 不一致；
2. finite mesh 不具备 $k\to k\pm q/2$ 平移不变性；
3. low temperature makes Fermi surface quadrature poorly converged；
4. density vertex 可能需要 orbital embedding phase，而不是单位矩阵；
5. contact expectation routing 可能不是 midpoint $f(H(k))$，需要 shifted 或 symmetrized thermal density matrix；
6. right Ward residual may require separate routing audit。

## 5. Boundary conclusion

本阶段不改公式，不回退 Stage 4.13 bubble sign fix，不调 residual，不加入 fitted
contact，不加入 $E^{ET}$，也不进入 conductivity / reflection / Casimir。

目标只是定位 $C-K$ 不等的来源。The remaining C-K mismatch should not be addressed
by reverting the Stage 4.13 bubble sign fix or changing direct-contact signs.
