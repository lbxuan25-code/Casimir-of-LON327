# Peierls Hamiltonian 顶角约定

## 用途

本文档记录 Peierls Hamiltonian vector vertex 与 Hamiltonian contact vertex 的顶角级推导和符号约定。它的职责只到
vertex-level：确认 Peierls 展开给出的 Hamiltonian vector vertex $V_i$ 和
Hamiltonian contact vertex $M_{ij}$。
它不是最终 conductivity、reflection 或 Casimir 结论，也不单独决定 response-level
physical current sign。

相关文档：

- [pipeline 总览](casimir_torque_response_pipeline_zh.md)
- [response-level Ward 符号推导](response_level_ward_convention_derivation_zh.md)
- [Ward 诊断结果汇总](ward_diagnostic_results_zh.md)

## hopping Hamiltonian

normal-state Hamiltonian 的 hopping/Fourier 表示写作

$$
H_0(\mathbf{k})=\sum_{\mathbf{R}} t_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} .
$$

该表示与现有三角函数 $H_0(\mathbf{k})$ 是同一个 Hamiltonian 的两种表达，不是新模型，也不是更高精度模型。所有 Peierls 顶角都必须从这组 $t_{\mathbf{R}}$ 出发。

## Peierls substitution

对 hopping bond 做 Peierls substitution：

$$
t_{\mathbf{R}}\rightarrow
t_{\mathbf{R}}\exp\!\left(i\int_{\mathrm{bond}}\mathbf{A}\cdot d\mathbf{l}\right).
$$

在 symmetric finite-q convention 下，一阶外场展开给出 Hamiltonian derivative vertex：

$$
V_i(\mathbf{k},\mathbf{q})
=
+i\sum_{\mathbf{R}}R_i t_{\mathbf{R}}e^{i\mathbf{k}\cdot\mathbf{R}}
\operatorname{sinc}\!\left(\frac{\mathbf{q}\cdot\mathbf{R}}{2}\right).
$$

其中

$$
\operatorname{sinc}(x)=\frac{\sin x}{x}.
$$

## Hamiltonian vector vertex 的 plus sign

Stage 4.1B 已验证顶角级 Ward identity：

$$
q_xV_x(\mathbf{k},\mathbf{q})
+q_yV_y(\mathbf{k},\mathbf{q})
=
H_0(\mathbf{k}+\mathbf{q}/2)-H_0(\mathbf{k}-\mathbf{q}/2).
$$

这说明 $+i\sum_R$ 的符号是 Hamiltonian vector vertex
$V_i=\delta H/\delta A_i$ 的正确顶角级符号。

代码对象对应为：

```text
peierls_hamiltonian_vector_vertex = V_i
```

旧的 `peierls_current_vertex(..., sign_convention=...)` 只保留为 historical sign-audit /
deprecated alias；主代码路径不再用 `sign_convention="plus"` 构造 $V_i$。

重要：这个 plus sign 不自动等同于 physical current 的符号。physical current 定义为

$$
j_i=-\frac{\delta H}{\delta A_i},
$$

因此 physical current vertex 是 $j_i=-V_i$。
不要把“顶角级 plus sign 通过”误读成“physical-current response 必须使用
$V_i$ 本身”。

## Hamiltonian contact vertex

Peierls phase 的二阶展开给出 Hamiltonian contact vertex：

$$
M_{ij}(\mathbf{k},\mathbf{q})
=
-\sum_{\mathbf{R}}R_iR_j t_{\mathbf{R}}e^{i\mathbf{k}\cdot\mathbf{R}}
\operatorname{sinc}^2\!\left(\frac{\mathbf{q}\cdot\mathbf{R}}{2}\right).
$$

负号来自

$$
\partial_i\partial_j e^{i\mathbf{k}\cdot\mathbf{R}}
=-R_iR_j e^{i\mathbf{k}\cdot\mathbf{R}}.
$$

因此 $q=0$ 时：

$$
M_{ij}(\mathbf{k},0)
=
\partial_i\partial_j H_0(\mathbf{k}).
$$

Stage 4.2A 已验证该 Hamiltonian contact vertex $M_{ij}$ 的 $q=0$ mass
limit、Hermiticity 和 $M_{xy}=M_{yx}$。

代码对象对应为：

```text
peierls_hamiltonian_contact_vertex = M_ij
```

它不是 physical direct contact contribution。physical-current response 中的标准对象是
$K_{ij}^{\mathrm{phys}}=-\langle M_{ij}\rangle$，其最终使用不由本文档单独决定，而由
[response-level Ward 符号推导](response_level_ward_convention_derivation_zh.md) 中的
physical current 定义、Kubo response convention 和 Ward diagnostic 共同约束。

## finite-q contact 与 q0 mass 的差异

由于

$$
\operatorname{sinc}^2(x)=1-\frac{x^2}{3}+O(x^4),
$$

finite-q contact 与 $q=0$ mass operator 的差异从 $O(q^2)$ 开始。因此它不能修复 response-level 中稳定存在的 $O(q)$ Ward residual。若 residual 的 leading error 是 $O(q)$，问题更可能来自 current/contact/physical response convention，而不是 $\operatorname{sinc}^2$ form factor。

## 当前结论

Peierls substitution 包括一阶 Hamiltonian vector vertex $V_i$ 和二阶
Hamiltonian contact vertex $M_{ij}$。$M_{ij}$ 只是 Peierls 展开的一部分，
不等于 physical direct contact contribution $K_{ij}^{\mathrm{phys}}$，也不等于整个 Peierls 方法。
当前顶角级结果支持 $V_i$ 和 $M_{ij}$ 本身基本可信；response-level
闭合还需要单独固定 physical current sign、contact response sign、Ward $q$-sign
和 Kubo convention。Residual minimization is not a physical derivation；residual
最小不能替代物理推导。
