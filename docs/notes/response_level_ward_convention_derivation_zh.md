# Response-level Ward convention 推导

## 用途

本文档保存 finite-q response 中 current sign、contact sign、Ward $q$-sign 与 Kubo response definition 的解析约定推导。它不是最终 conductivity、reflection 或 Casimir 实现，也不是材料结论。

相关文档：

- [pipeline 总览](casimir_torque_response_pipeline_zh.md)
- [Peierls 顶角约定](peierls_vertex_convention_zh.md)
- [Ward 诊断结果汇总](ward_diagnostic_results_zh.md)

## A. Hamiltonian coupling

从外场耦合 Hamiltonian 写起：

$$
H[\mathbf{A}]
=
H_0
+A_i\Gamma_i^H
+\frac{1}{2}A_i\Lambda_{ij}^H A_j+\cdots .
$$

其中

$$
\Gamma_i^H=\frac{\delta H}{\delta A_i},\qquad
\Lambda_{ij}^H=\frac{\delta^2 H}{\delta A_i\delta A_j}.
$$

$\Gamma_i^H$ 是 Hamiltonian derivative current vertex；$\Lambda_{ij}^H$ 是 Hamiltonian second derivative/contact vertex。

## B. Hamiltonian derivative vertex 与 physical current

physical current 定义为

$$
j_i=-\frac{\delta H}{\delta A_i}.
$$

代入上式：

$$
j_i
=
-\Gamma_i^H-\Lambda_{ij}^H A_j+\cdots .
$$

因此 physical current vertex 是

$$
\Gamma_i^{\mathrm{phys}}=-\Gamma_i^H,
$$

而不是 $\Gamma_i^H$ 本身。顶角级 Peierls plus sign 只说明 $\Gamma_i^H$ 的 Hamiltonian derivative convention 正确；它不自动决定 physical current response 的符号。

## C. 单粒子 Ward identity

定义

$$
G^{-1}(\mathbf{k},i\omega)=i\omega-H_0(\mathbf{k}).
$$

令

$$
k_+=(\mathbf{k}+\mathbf{q}/2,i\omega+i\Omega/2),
\qquad
k_-=(\mathbf{k}-\mathbf{q}/2,i\omega-i\Omega/2).
$$

则

$$
G^{-1}(k_+)-G^{-1}(k_-)
=
i\Omega I-\left[
H_0(\mathbf{k}+\mathbf{q}/2)-H_0(\mathbf{k}-\mathbf{q}/2)
\right].
$$

利用 Peierls 顶角级 identity：

$$
H_0(\mathbf{k}+\mathbf{q}/2)-H_0(\mathbf{k}-\mathbf{q}/2)
=q_i\Gamma_i^H(\mathbf{k},\mathbf{q}),
$$

得到

$$
G^{-1}(k_+)-G^{-1}(k_-)
=
i\Omega\Gamma_0-q_i\Gamma_i^H .
$$

这里 $\Gamma_0=I$ 是当前 prototype density vertex。

## D. 两套自洽 convention

### Hamiltonian-vertex convention

若 spatial vertex 使用 Hamiltonian derivative vertex：

$$
\Gamma_i=\Gamma_i^H,
$$

则 Ward contraction 应使用

$$
Q_H=(i\Omega,-q_x,-q_y).
$$

也就是

$$
i\Omega\Pi_{0\nu}-q_x\Pi_{x\nu}-q_y\Pi_{y\nu}=0,
$$

以及右 contraction 的相应形式。

### Physical-current convention

若 spatial current observable 使用 physical current：

$$
j_i=-\Gamma_i^H,
$$

则可使用

$$
Q_{\mathrm{phys}}=(i\Omega,+q_x,+q_y),
$$

也就是

$$
i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu}=0.
$$

两套 convention 都可以成立，但不能混用。

## E. 当前最可疑的 mismatch

当前代码中的可疑点是：

- spatial vertex 使用 Peierls plus $\Gamma_i^H$；
- Ward residual 使用 $+q_x,+q_y$ 的 physical-current-like contraction。

这相当于把 Hamiltonian derivative vertex 与 physical-current Ward $q$-sign 混在一起，可能自然留下 spatial residual 的 $O(q)$ 项。

## F. Contact sign

从

$$
j_i=-\frac{\delta H}{\delta A_i}
=-\Gamma_i^H-\Lambda_{ij}^H A_j
$$

可得 direct contact response：

$$
\frac{\delta j_i}{\delta A_j}=-\Lambda_{ij}^H .
$$

因此在 physical-current convention 下，contact direct term 倾向于以

$$
-\langle\Lambda_{ij}^H\rangle
$$

进入 spatial response。已有 spatial raw residual 中 contact minus sign 更好，这与该解析方向一致。

但这仍需 response-level 数值验证。不能仅凭 residual 直接改最终 conductivity 代码；还必须同时固定 bubble sign、equal-time term、指标顺序和 external-field convention。

## G. density sector 为什么是 O(q²)

density-current block 在 small $q$ 下通常自身为 $O(q)$。因此 density residual 中

$$
q_i\Pi_{i0}
$$

常表现为 $O(q^2)$。Stage 4.3B 的 density-current audit 也显示 density residual 的 small-q scaling $\alpha\approx2$，且对 contact scheme 不敏感。所以 density sector 不是 full Ward $O(q)$ 主因。

## H. spatial sector 为什么是 O(q)

spatial-spatial block $\Pi_{ij}$ 在 $q\to0$ 可以有限。若 current sign 或 Ward $q$-sign 混用，则

$$
q_i\Pi_{ij}
$$

直接留下 $O(q)$ residual。Stage 4.3C 的 spatial-current audit 显示 spatial residual 的 small-q scaling $\alpha\approx1$，与该判断一致。

## I. 下一步数值验证

下一步不应继续盲调 Peierls form factor，而应比较两套 self-consistent convention。

Hamiltonian-vertex convention：

- current vertex $=\Gamma_i^H$；
- Ward $q$-sign 使用 $Q_H=(i\Omega,-q_x,-q_y)$；
- contact sign 以 Hamiltonian response convention 的 plus candidate 为基准。

Physical-current convention：

- current observable $=-\Gamma_i^H$；
- Ward $q$-sign 使用 $Q_{\mathrm{phys}}=(i\Omega,+q_x,+q_y)$；
- contact direct term倾向于 minus candidate，即 $-\Lambda_{ij}^H$。

## 当前结论

本文档是解析符号约定推导，不是最终 conductivity/reflection/Casimir 实现。当前最可疑的是 response-level convention mismatch：Hamiltonian derivative current vertex、physical current、contact sign、Ward $q$-sign 与 Kubo response definition 尚未统一。
