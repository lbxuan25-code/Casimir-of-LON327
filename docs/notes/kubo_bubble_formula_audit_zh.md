# Kubo bubble formula audit

## 用途

本文档审计 normal-state finite-q physical-current response candidate 中的 Kubo
bubble 公式。它只检查 response-level Kubo convention，不是 residual fitting，不是
finite-q conductivity、reflection 或 Casimir 输入，也不声明 Ward identity 已闭合。

相关文档：

- [pipeline 总览](casimir_torque_response_pipeline_zh.md)
- [response-level Ward convention 推导](response_level_ward_convention_derivation_zh.md)
- [equal-time / commutator Ward 审计](equal_time_commutator_ward_audit_zh.md)
- [Ward 诊断结果汇总](ward_diagnostic_results_zh.md)

## Conventions

外场耦合写成

$$
H[a]=H_0+a_\nu P_\nu+\frac12 A_iM_{ij}A_j+\cdots ,
$$

其中

$$
a_\nu=(\phi,A_x,A_y).
$$

source-coupling vertices 定义为

$$
P_0=\rho,\qquad P_i=V_i,
$$

其中 $V_i=\delta H/\delta A_i$。

physical observable vertices 定义为

$$
J_0=\rho,\qquad J_i=j_i=-V_i.
$$

physical response 定义为

$$
\Pi_{\mu\nu}(i\Omega,\mathbf q)
=
\frac{\delta\langle J_\mu\rangle}{\delta a_\nu}.
$$

imaginary-time linear response 给出

$$
\frac{\delta\langle J_\mu\rangle}{\delta a_\nu}
=
-\int_0^\beta d\tau\,
\langle T_\tau J_\mu(\tau)P_\nu(0)\rangle_c
+
\left\langle
\frac{\delta J_\mu}{\delta a_\nu}
\right\rangle .
$$

重点：bubble 的第二个 vertex 是 source-coupling vertex $P_\nu$，不是自动等于
observable vertex $J_\nu$。spatial-spatial direct term 为

$$
\left\langle
\frac{\delta J_i}{\delta A_j}
\right\rangle
=
-\langle M_{ij}\rangle .
$$

## Finite-q band-sum

有限动量下使用 shifted momenta

$$
\mathbf k_-=\mathbf k-\mathbf q/2,\qquad
\mathbf k_+=\mathbf k+\mathbf q/2.
$$

令

$$
H_-=H_0(\mathbf k_-),\qquad H_+=H_0(\mathbf k_+),
$$

$$
H_-|m,-\rangle=E_m^-|m,-\rangle,\qquad
H_+|n,+\rangle=E_n^+|n,+\rangle.
$$

定义 matrix element

$$
X_{mn}^{-+}=\langle m,-|X(\mathbf k,\mathbf q)|n,+\rangle .
$$

source-side reverse element 应为

$$
Y_{nm}^{+-}=\langle n,+|Y(\mathbf k,-\mathbf q)|m,-\rangle .
$$

当前 Peierls prototype 满足 finite-q Hermiticity relation

$$
Y(\mathbf k,\mathbf q)^\dagger=Y(\mathbf k,-\mathbf q),
$$

对 $\rho$、$V_i$ 和 $M_{ij}$ 在本模型中成立。因此代码可以用

$$
Y_{nm}^{+-}=\left(Y_{mn}^{-+}\right)^*
$$

实现 reverse matrix element。该关系由
`tests/test_kubo_bubble_formula_audit.py` 中的 finite-q reverse-element test 覆盖。

Matsubara sum 使用 routing

$$
\frac{1}{\beta}\sum_{i\omega}
\frac{1}{i\omega+i\Omega-E_n^+}
\frac{1}{i\omega-E_m^-}
=
\frac{f(E_m^-)-f(E_n^+)}
{i\Omega+E_m^- -E_n^+}.
$$

再乘上 response definition 中的整体负号，bubble factor 为

$$
\mathrm{factor}_{mn}
=
-\frac{f(E_m^-)-f(E_n^+)}
{i\Omega+E_m^- -E_n^+}.
$$

因此：

- numerator 使用 $f_m^- - f_n^+$；
- denominator 使用 $i\Omega+E_m^- -E_n^+$；
- 前面有来自 $\delta\langle J\rangle/\delta a$ 的整体负号；
- 当前代码中的 `factor = -occupation_diff / denominator` 与该推导匹配。

## Observable vertex vs source-coupling vertex

由定义

$$
\Pi_{\mu\nu}=\frac{\delta\langle J_\mu\rangle}{\delta a_\nu}
$$

可知 bubble 必须是

$$
-\langle J_\mu P_\nu\rangle,
$$

而不是一般的 $-\langle J_\mu J_\nu\rangle$。对 spatial source，

$$
P_i=V_i=-j_i.
$$

因此 physical-current response candidate 的 bubble block 应直接读作

$$
\Pi_{00}^{\mathrm{bubble}}=\mathrm{bubble}[\rho,\rho],
$$

$$
\Pi_{0j}^{\mathrm{bubble}}=\mathrm{bubble}[\rho,V_j],
$$

$$
\Pi_{i0}^{\mathrm{bubble}}=\mathrm{bubble}[-V_i,\rho],
$$

$$
\Pi_{ij}^{\mathrm{bubble}}=\mathrm{bubble}[-V_i,V_j].
$$

spatial-spatial direct term 再加入

$$
\Pi_{ij}^{\mathrm{direct}}=-\langle M_{ij}\rangle.
$$

Stage 4.7 版本的 readable physical response candidate 曾在 bubble 两侧都使用
`vertices=(rho,jx,jy)`，等价于 $\mathrm{bubble}[J_\mu,J_\nu]$。这与上面的
source-coupling convention 不匹配。Stage 4.8 已修复为：

```python
observable_vertices = (rho, jx, jy)
source_vertices = (rho, vector_x, vector_y)
```

并由私有 helper `_finite_q_band_bubble_imag_axis(...)` 统一实现 band-sum bubble。

## Code comparison table

| 项目 | 解析推导 | 当前代码 | 结论 |
|---|---|---|---|
| response 定义 | $\delta\langle J_\mu\rangle/\delta a_\nu$ | `normal_physical_density_current_response_imag_axis` 文档和实现均按 observable/source 分离 | MATCH |
| observable vertex | $J_0=\rho,\ J_i=-V_i$ | `observable_vertices = (rho, jx, jy)` | MATCH |
| source vertex | $P_0=\rho,\ P_i=V_i$ | `source_vertices = (rho, vector_x, vector_y)` | MATCH |
| overall sign | response definition 给出 bubble 前整体负号 | `factor = -occupation_diff / denominator` | MATCH |
| occupation order | $f(E_m^-)-f(E_n^+)$ | `occupations_minus[m] - occupations_plus[n]` | MATCH |
| denominator | $i\Omega+E_m^- -E_n^+$ | `1j * omega + energy_minus - energy_plus` | MATCH |
| matrix element order | $J_{\mu,mn}^{-+}P_{\nu,nm}^{+-}$ | `observable[m,n] * conj(source[m,n])`，依赖 finite-q Hermiticity relation | MATCH |
| contact / equal-time | direct term $-\langle M_{ij}\rangle$；是否还需要额外 commutator/equal-time completion 需另行证明 | `physical_direct_contact = -expect_mij` 已实现 direct term | UNRESOLVED |

`UNRESOLVED` 的具体含义：当前 direct contact term 与
$\langle\delta J_i/\delta A_j\rangle=-\langle M_{ij}\rangle$ 匹配，但本文档没有证明
Kubo response 在本项目的 density/current convention 下不再需要额外 equal-time /
commutator completion。因此不能声明 gauge closure 或 finite-q conductivity 已完成。

## Current code status after audit

Stage 4.8 的明确结论：

1. bubble factor 的整体负号、occupation difference 顺序和 denominator 顺序为 `MATCH`；
2. finite-q matrix element product order 在当前 Peierls Hermiticity relation 下为 `MATCH`；
3. Stage 4.7 physical response candidate 的 source-side vertex 为 `MISMATCH`，已在 Stage 4.8 修复；
4. 修复后 bubble 使用 $\mathrm{bubble}[J_\mu,P_\nu]$；
5. direct contact $-\langle M_{ij}\rangle$ 为 `MATCH`；
6. 是否还缺额外 equal-time / commutator correction 为 `UNRESOLVED`。

本审计没有使用 residual minimization 决定公式，没有引入自由 sign 参数，没有接入
conductivity、reflection 或 Casimir，也没有声明 Ward identity 已闭合。
