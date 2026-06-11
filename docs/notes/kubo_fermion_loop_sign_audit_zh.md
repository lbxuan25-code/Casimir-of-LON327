# Kubo bubble fermion-loop sign audit

## 1. Source coupling and response definition

本审计只讨论 normal-state response-level bubble 的 overall sign，不修改主路径。
从外源耦合

$$
H[a]=H_0+a_\nu P_\nu
$$

开始，response 定义为

$$
\Pi_{\mu\nu}=\frac{\delta\langle J_\mu\rangle}{\delta a_\nu}.
$$

imaginary-time 线性响应给出

$$
\Pi_{\mu\nu}^{\mathrm{bubble}}
=
-\left\langle T_\tau J_\mu P_\nu\right\rangle_c .
$$

这里的负号来自对 $H[a]=H_0+aP$ 的 source perturbation 做线性响应。

## 2. Fermion loop sign

对二次型算符

$$
O_A=c^\dagger A c,\qquad O_B=c^\dagger B c,
$$

采用 Matsubara Green function convention

$$
G_{ab}(\tau)=-\langle T_\tau c_a(\tau)c_b^\dagger(0)\rangle.
$$

Wick contraction 的 connected part 为

$$
\left\langle T_\tau O_A(\tau)O_B(0)\right\rangle_c
=
-\operatorname{Tr}\left[A\,G(\tau)\,B\,G(-\tau)\right].
$$

因此

$$
\Pi^{\mathrm{bubble}}
=
-\langle T_\tau O_AO_B\rangle_c
=
+\operatorname{Tr}[A G B G].
$$

结论是

$$
\boxed{\text{linear-response minus sign and fermion-loop minus sign cancel.}}
$$

## 3. Band-sum sign

finite-$q$ band basis 中定义

$$
k_\pm=k\pm q/2.
$$

Matsubara sum 为

$$
\frac{1}{\beta}
\sum_{i\omega}
\frac{1}{i\omega+i\Omega-E_n^+}
\frac{1}{i\omega-E_m^-}
=
\frac{f(E_m^-)-f(E_n^+)}
{i\Omega+E_m^- -E_n^+}.
$$

因此 candidate physical bubble 为

$$
\boxed{
\Pi_{\mu\nu}^{\mathrm{bubble},+}
=
\sum_{k,m,n}
\frac{f(E_m^-)-f(E_n^+)}
{i\Omega+E_m^- -E_n^+}
J_{\mu,mn}^{-+}P_{\nu,nm}^{+-}
}.
$$

当前实现中的 bubble 等价于在同一 band matrix element convention 上额外乘了 overall
minus sign：

$$
\Pi_{\mu\nu}^{\mathrm{bubble,current}}
=
-\sum_{k,m,n}
\frac{f(E_m^-)-f(E_n^+)}
{i\Omega+E_m^- -E_n^+}
J_{\mu,mn}^{-+}P_{\nu,nm}^{+-}.
$$

Stage 4.12 只审计这一个 overall sign，不把 positive candidate 接入主 response path。

## 4. Ward contraction sign

当前 physical-current convention 为

$$
J_0=\rho,\qquad J_i=-V_i.
$$

Peierls vertex identity 为

$$
q_iV_i(k,q)=H_+(k)-H_-(k).
$$

因此

$$
i\Omega J_0+q_iJ_i
=
i\Omega\rho-q_iV_i.
$$

在 band matrix element 中，

$$
\langle m,-|i\Omega\rho-q_iV_i|n,+\rangle
=
(i\Omega+E_m^- -E_n^+)\rho_{mn}^{-+}.
$$

所以 positive bubble candidate 应满足

$$
R^{\mathrm{bubble},+}_{L,\nu}=+C_\nu,
$$

而 current negative bubble 会满足

$$
R^{\mathrm{bubble,current}}_{L,\nu}=-C_\nu.
$$

## 5. Compressibility sanity check

令

$$
H=H_0+aN,\qquad P=N,\qquad J=N.
$$

真实物理响应为

$$
\frac{\delta\langle N\rangle}{\delta a}
=
\sum_n f'(E_n)<0.
$$

其中

$$
f'(E)=-\frac{1}{T}f(E)(1-f(E)).
$$

positive bubble 在静态极限给出 $f'(E)<0$；negative bubble 给出
$-f'(E)>0$。因此 compressibility sanity check 支持 positive band-sum
sign。

## 6. 结论边界

- 本文档不直接修改主路径。
- analytic audit suggests current bubble overall sign may be reversed。
- numerical diagnostic is required before changing main response。
- direct contact sign is not questioned here。
- conductivity / reflection / Casimir 仍未接入。
- 本审计不声明 Ward identity closure。
