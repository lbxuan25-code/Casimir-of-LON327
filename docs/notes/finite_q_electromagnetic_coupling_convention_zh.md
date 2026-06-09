# finite-q 电磁耦合约定

## 1. Purpose / 目的

本文记录 finite-q 响应中 Hamiltonian 表示、电磁顶角、Ward identity 与 contact term
的使用边界，避免把不同层次的诊断混在一起。

三角函数形式和 hopping/Fourier 指数形式是同一个 $H_0(\mathbf{k})$ 的两种等价表示。
hopping/Fourier 表示不是新模型，不是更高精度模型；它只是为了从 Peierls 相位系统地构造
finite-q 电磁耦合顶角。

## 2. Hamiltonian representations

当前 `src/lno327/model.py` 中的参考实现记为

$$
H_0^{\mathrm{trig}}(\mathbf{k}) .
$$

等价的 hopping/Fourier 表示写作

$$
H_0^{\mathrm{hop}}(\mathbf{k}) = \sum_{\mathbf{R}} t_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} .
$$

任何 hopping/Fourier 实现都必须先通过重构验证：

$$
H_0^{\mathrm{hop}}(\mathbf{k}) = H_0^{\mathrm{trig}}(\mathbf{k})
$$

在声明的数值容差内成立。否则不能用它构造后续 finite-q 电磁顶角。

## 3. Scope / 使用边界

- 能带、本征值、本征矢、$q=0$ local response、已有 local conductivity 可以继续使用
  三角函数形式和 $\partial H_0/\partial k_i$ 顶角。
- finite-q Ward diagnostic、未来 gauge-consistent
  $\Pi_{\mu\nu}(i\omega,\mathbf{q})$、未来 finite-q conductivity/reflection/Casimir
  input 必须使用 Peierls-consistent 顶角。
- 现有 Stage 1 / Stage 3 current-current finite-q kernel 仍是 diagnostic-only，
  不会自动升级为 gauge-closed conductivity。

## 4. Density vertex convention

当前 prototype 约定使用

$$
\Gamma_0 = I_4 .
$$

这等价于忽略 in-plane orbital embedding，或把四个轨道视为共享同一个 in-plane 位置。
未来若加入 orbital embedding，应改为

$$
[\Gamma_0(\mathbf{q})]_{ab} = \delta_{ab} e^{-i\mathbf{q}\cdot\mathbf{r}_a}.
$$

## 5. Current vertex convention

midpoint velocity

$$
v_i(\mathbf{k}) = \frac{\partial H_0}{\partial k_i}
$$

是 $q\to0$ 极限，可以作为 diagnostic baseline，但不是最终 finite-q gauge-consistent
current vertex。

从 hopping/Fourier 表示的 Peierls 相位展开得到的 finite-q current vertex 约定为

$$
\Gamma_i^P(\mathbf{k},\mathbf{q})
= i \sum_{\mathbf{R}} R_i t_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}}
\,\operatorname{sinc}\!\left(\frac{\mathbf{q}\cdot\mathbf{R}}{2}\right),
$$

其中

$$
\operatorname{sinc}(x)=\frac{\sin x}{x}.
$$

实现时应检查 vertex-level Ward identity：

$$
q_x \Gamma_x^P + q_y \Gamma_y^P
= H_0(\mathbf{k}+\mathbf{q}/2)-H_0(\mathbf{k}-\mathbf{q}/2).
$$

整体符号可由项目中的电荷/外场约定固定，但必须在实现和输出 metadata 中明确记录。

## 6. Contact / diamagnetic term convention

spatial-spatial response 除 current-current bubble 外，还需要 contact / diamagnetic term。

$q=0$ mass operator

$$
\frac{\partial^2 H_0}{\partial k_i \partial k_j}
$$

只能作为 small-q diagnostic contact。最终 finite-q gauge-consistent response 需要从同一个
Peierls 展开得到 finite-q contact term。没有 contact term 的
$\Pi_{\mu\nu}$ 不能称为最终 finite-q conductivity。

## 7. Ward diagnostic convention

Ward diagnostic 使用

$$
\Pi_{\mu\nu}(i\Omega_n,\mathbf{q}), \qquad \mu,\nu=0,x,y,
$$

其中 $0$ 表示 density，$x/y$ 表示 current。残差定义为

$$
R_L[\nu] = i\Omega\,\Pi_{0\nu} + q_x\Pi_{x\nu} + q_y\Pi_{y\nu},
$$

$$
R_R[\mu] = i\Omega\,\Pi_{\mu0} + \Pi_{\mu x}q_x + \Pi_{\mu y}q_y.
$$

只有 Ward residual 在声明的 vertex/contact scheme 下足够小，才能称为
gauge-consistent finite-q response。

## 8. Required provenance fields

后续 finite-q response 的 summary / compact CSV 至少记录：

```text
hamiltonian_representation
vertex_scheme
contact_scheme
density_vertex_scheme
ward_identity_checked
gauge_closed
conductivity_computed
casimir_computed
not_final_casimir_conclusion
```
