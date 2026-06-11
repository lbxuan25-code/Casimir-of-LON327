# Stage 4.13 bubble prefactor sign fix

## 1. 为什么要改

Stage 4.12 的解析和数值审计都指向同一问题：current main response path 中
finite-q Kubo bubble 的 overall sign reversed。

解析上，

$$
\Pi^{\mathrm{bubble}}=-\langle TJP\rangle_c,
$$

而 fermion bilinear connected correlator 带有 fermion-loop 负号：

$$
\langle TJP\rangle_c=-\mathrm{Tr}[JGPG].
$$

因此

$$
\Pi^{\mathrm{bubble}}=+\mathrm{Tr}[JGPG].
$$

Stage 4.12 数值诊断也显示 positive bubble candidate 给
$R_L^{\mathrm{bubble}}\approx +C_j$，而 old negative bubble 给
$R_L^{\mathrm{bubble}}\approx -C_j$。

## 2. 改了什么

Stage 4.13 只翻转 `_finite_q_band_bubble_imag_axis(...)` 的整体 prefactor：

$$
-\frac{f(E_m^-)-f(E_n^+)}
{i\Omega+E_m^- -E_n^+}
\quad\longrightarrow\quad
+\frac{f(E_m^-)-f(E_n^+)}
{i\Omega+E_m^- -E_n^+}.
$$

修正后的 bubble 公式为

$$
\Pi_{\mu\nu}^{\mathrm{bubble}}
=
\sum_{k,m,n}
\frac{
f(E_m^-)-f(E_n^+)
}{
i\Omega+E_m^- -E_n^+
}
J_{\mu,mn}^{-+}
P_{\nu,nm}^{+-}.
$$

## 3. 没改什么

本阶段没有修改：

- $V_i=\delta H/\delta A_i$；
- $M_{ij}=\delta^2H/\delta A_i\delta A_j$；
- physical current $j_i=-V_i$；
- source/observable split
  $J=(\rho,-V_x,-V_y)$、$P=(\rho,V_x,V_y)$；
- direct contact $D_{ij}=-\langle M_{ij}\rangle$；
- conductivity / reflection / Casimir 相关代码。

没有加入 fitted contact，没有加入 $E^{ET}$，没有做 residual tuning。

## 4. 修正后预期

修正后的符号 bookkeeping 应满足

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

如果 $C_j-K_j$ 仍不为零，下一阶段应审计 density/source $q$-convention、
commutator routing、contact thermal expectation，而不是回退 bubble sign。

本阶段不声明 Ward identity closure，也不产生 finite-q conductivity 或 Casimir 结论。
