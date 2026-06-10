# Response-level Ward convention 推导

## 用途

本文档保存 finite-q response 中 current sign、contact sign、Ward $q$-sign 与 Kubo response definition 的解析约定推导。它不是最终 conductivity、reflection 或 Casimir 实现，也不是材料结论。

相关文档：

- [pipeline 总览](casimir_torque_response_pipeline_zh.md)
- [Peierls 顶角约定](peierls_vertex_convention_zh.md)
- [Ward 诊断结果汇总](ward_diagnostic_results_zh.md)

## Standard notation / 标准命名

本节固定 response-level 文档中的标准对象名。后续文档应避免裸写 “current vertex”、
“contact term” 或 “direct contact”；第一次出现时应写出下列标准名。

| 标准名 | 定义 | 代码函数 / 代码对象 | 使用边界 |
|---|---|---|---|
| Hamiltonian vector vertex $\Gamma_i^H$ | $\Gamma_i^H=\delta H/\delta A_i$ | `peierls_current_vertex(sign_convention="plus")` | Hamiltonian 展开的一阶 vertex；不要把它叫作 physical current vertex |
| physical current vertex $\Gamma_i^{\mathrm{phys}}$ | $\Gamma_i^{\mathrm{phys}}=j_i^{(0)}=-\Gamma_i^H$ | 不是当前 Peierls 函数的直接返回值 | 只在讨论 physical current observable 时使用 |
| Hamiltonian contact vertex $\Lambda_{ij}^H$ | $\Lambda_{ij}^H=\delta^2H/\delta A_i\delta A_j$ | `peierls_contact_vertex` | Hamiltonian 展开的二阶 vertex；不要把它直接叫作 physical direct contact |
| code contact extraction $C_{ij}^{\mathrm{code}}$ | $C_{ij}^{\mathrm{code}}=\texttt{contact\_only}=+\langle\Lambda_{ij}^H\rangle$ | `response(finite_q_peierls, plus) - response(none)` | 代码 plus-contact extraction；不是 physical direct contact contribution 本身 |
| physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ | $K_{ij}^{\mathrm{phys}}=-\langle\Lambda_{ij}^H\rangle=-C_{ij}^{\mathrm{code}}$ | 由 physical current definition 推出 | physical-current response 中的 direct / diamagnetic / contact contribution |
| code bubble prototype $B_{\mu\nu}^{\mathrm{code}}$ | $B_{\mu\nu}^{\mathrm{code}}=\Pi_{\mu\nu}^{\mathrm{code}}(\texttt{contact\_scheme=none})$ | `normal_density_current_response_imag_axis(..., contact_scheme="none")` | 当前 Kubo bubble prototype；不是已确认的 final physical response |
| Hamiltonian Ward vector $Q_H$ | $Q_H=(i\Omega,-q_x,-q_y)$ | diagnostic contraction choice | 与 Hamiltonian vector vertex $\Gamma_i^H$ 配套 |
| physical Ward vector $Q_{\mathrm{phys}}$ | $Q_{\mathrm{phys}}=(i\Omega,+q_x,+q_y)$ | diagnostic contraction choice | 与 physical current vertex $\Gamma_i^{\mathrm{phys}}$ 配套 |

Hamiltonian vector vertex 的代码公式为

$$
\Gamma_i^H(k,q)
=
+i\sum_R R_i t_R e^{ik\cdot R}
\operatorname{sinc}\!\left(\frac{q\cdot R}{2}\right).
$$

Hamiltonian contact vertex 的代码公式为

$$
\Lambda_{ij}^H(k,q)
=
-\sum_R R_iR_jt_R e^{ik\cdot R}
\operatorname{sinc}^2\!\left(\frac{q\cdot R}{2}\right).
$$

code bubble prototype 到 physical-current bubble candidate 的符号变换可写成

$$
S=\mathrm{diag}(1,-1,-1),
$$

$$
B_{\mu\nu}^{\mathrm{phys,candidate}}
=
S_{\mu\mu'}B_{\mu'\nu'}^{\mathrm{code}}S_{\nu'\nu}.
$$

因此 spatial-spatial block 因两个 physical current vertex
$\Gamma_i^{\mathrm{phys}}$ 符号相乘而不变；density-current 和 current-density block
因只有一个 physical current vertex $\Gamma_i^{\mathrm{phys}}$ 而变号。

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

$\Gamma_i^H$ 是 Hamiltonian vector vertex；$\Lambda_{ij}^H$ 是 Hamiltonian contact vertex。

## B. Hamiltonian vector vertex 与 physical current vertex

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

而不是 $\Gamma_i^H$ 本身。顶角级 Peierls plus sign 只说明 Hamiltonian vector
vertex $\Gamma_i^H$ 的符号正确；它不自动决定 physical-current response 中
$\Gamma_i^{\mathrm{phys}}$ 的符号使用。

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

可得 physical direct contact contribution 的 vertex-level 来源：

$$
\frac{\delta j_i}{\delta A_j}=-\Lambda_{ij}^H .
$$

因此在 physical-current convention 下，contact direct term 应以

$$
-\langle\Lambda_{ij}^H\rangle
$$

进入 spatial response。已有 spatial raw residual 中 contact-minus candidate 更好，这与该解析方向一致。

但这仍需 response-level diagnostic 继续复查。Residual minimization is not a physical
derivation；residual 最小不能替代物理推导。不能仅凭 residual 直接改最终
conductivity 代码；还必须同时固定 bubble sign、equal-time term、指标顺序和
external-field convention。

## G. 代码对象与解析对象对应 / Code objects and analytic objects

本节把当前代码中的对象与解析公式逐一对应。这里的“可以确定”只表示
formula-to-code mapping 已清楚；它不表示 response-level Ward identity 已经闭合。

| 代码函数 / 代码对象 | 标准名 | 数学定义 | 物理层级 | 是否 final physical response |
|---|---|---|---|---|
| `peierls_current_vertex(sign_convention="plus")` | Hamiltonian vector vertex $\Gamma_i^H$ | $+i\sum_R R_i t_R e^{i k\cdot R}\operatorname{sinc}\!\left(\frac{q\cdot R}{2}\right)$ | Hamiltonian 展开的一阶 vertex，不是 physical current vertex $\Gamma_i^{\mathrm{phys}}$ | 否；顶角级 Ward identity 已验证 |
| `peierls_contact_vertex` | Hamiltonian contact vertex $\Lambda_{ij}^H$ | $-\sum_R R_iR_j t_R e^{ik\cdot R}\operatorname{sinc}^2\!\left(\frac{q\cdot R}{2}\right)$ | Hamiltonian 展开的二阶 vertex，不是 $K_{ij}^{\mathrm{phys}}$ | 否；$q=0$ mass limit、Hermiticity、$\Lambda_{xy}=\Lambda_{yx}$ 已验证 |
| `normal_density_current_response_imag_axis(..., contact_scheme="none")` / `response_none` | code bubble prototype $B_{\mu\nu}^{\mathrm{code}}$ | $\Pi_{\mu\nu}^{\mathrm{code}}(\texttt{contact\_scheme=none})$ | 用 $\Gamma_0=I$ 与 Hamiltonian vector vertex $\Gamma_i^H$ 构造的 Kubo bubble prototype | 否；可用于 diagnostic，不能直接称为 finite-q conductivity |
| spatial-spatial block of `response_none` | $B_{ij}^{\mathrm{code}}$ | $B_{ij}^{\mathrm{code}}\sim\langle\Gamma_i^H\Gamma_j^H\rangle_{\mathrm{Kubo}}$ | physical-current bubble candidate 中两个 $\Gamma_i^{\mathrm{phys}}=-\Gamma_i^H$ 相乘，spatial-spatial block 整体符号不变 | 否；Kubo convention 仍需闭合 |
| density-current / current-density block of `response_none` | $B_{0i}^{\mathrm{code}}$ / $B_{i0}^{\mathrm{code}}$ | $B_{0i}^{\mathrm{code}}\sim\langle\Gamma_0\Gamma_i^H\rangle$，$B_{i0}^{\mathrm{code}}\sim\langle\Gamma_i^H\Gamma_0\rangle$ | physical-current bubble candidate 中只有一个 spatial vertex，因此这些 block 应随 $\Gamma_i^{\mathrm{phys}}=-\Gamma_i^H$ 变号 | 否；是 response-level sign diagnostic 的关键 |
| `contact_only = response(finite_q_peierls, plus) - response(none)` | code contact extraction $C_{ij}^{\mathrm{code}}$ | $C_{ij}^{\mathrm{code}}=+\sum_{n,k}w_k f_{n,k}\langle n,k|\Lambda_{ij}^H(k,q)|n,k\rangle=+\langle\Lambda_{ij}^H\rangle$ | 代码 plus-contact extraction；不是 physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ | 否；代码对象明确，不是未知项 |
| physical-current direct contact | physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ | $K_{ij}^{\mathrm{phys}}=-\langle\Lambda_{ij}^H\rangle=-C_{ij}^{\mathrm{code}}$ | 从 $j_i=-\delta H/\delta A_i=-\Gamma_i^H-\Lambda_{ij}^HA_j+\cdots$ 得出 | 否；是 `physical_current_q_plus_contact_minus` 的解析动机 |

`peierls_current_vertex(sign_convention="plus")` 满足顶角级 Ward identity：

$$
q_i\Gamma_i^H(k,q)
=
H_0(k+q/2)-H_0(k-q/2).
$$

physical current vertex 是

$$
j_i^{(0)}=-\Gamma_i^H.
$$

从

$$
H[A]=H_0+A_i\Gamma_i^H+\frac12 A_i\Lambda_{ij}^H A_j+\cdots
$$

得到

$$
j_i=-\frac{\delta H}{\delta A_i}
=-\Gamma_i^H-\Lambda_{ij}^H A_j+\cdots,
$$

因此

$$
\frac{\delta j_i}{\delta A_j}=-\Lambda_{ij}^H.
$$

如果 response 定义为 physical response

$$
\Pi_{ij}^{\mathrm{phys}}=\frac{\delta\langle j_i\rangle}{\delta A_j},
$$

那么 physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ 应为

$$
-\langle\Lambda_{ij}^H\rangle.
$$

这就是 Stage 4.4 / 4.5 中 `physical_current_q_plus_contact_minus` 有解析动机的原因：
它不是纯粹数值拟合出来的符号选择。但它仍只是 best residual candidate /
best diagnostic candidate，不是最终 convention 或最终物理实现，因为 spatial Ward
residual 仍为 $O(q)$，Ward identity 尚未闭合。

当前可以确定：

1. `peierls_current_vertex(sign_convention="plus")` 是 $\Gamma_i^H$。
2. `peierls_contact_vertex` 是 $\Lambda_{ij}^H$。
3. `contact_only` 在代码 plus-contact 抽取下是 code contact extraction
   $C_{ij}^{\mathrm{code}}=+\langle\Lambda_{ij}^H\rangle$。
4. physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ 应为
   $-\langle\Lambda_{ij}^H\rangle=-C_{ij}^{\mathrm{code}}$。
5. contact-minus candidate 有解析动机，并且降低 residual。

当前不能确定：

1. 当前 prototype $\Pi$ 是否已经是完整 physical response。
2. Kubo bubble 的整体符号、denominator、matrix-element order 是否完全正确。
3. 是否还缺 equal-time / commutator term。
4. Ward identity 是否已经闭合。
5. finite-q conductivity / reflection / Casimir 是否可以接入。

## H. density sector 为什么是 O(q²)

density-current block 在 small $q$ 下通常自身为 $O(q)$。因此 density residual 中

$$
q_i\Pi_{i0}
$$

常表现为 $O(q^2)$。Stage 4.3B 的 density-current audit 也显示 density residual 的 small-q scaling $\alpha\approx2$，且对 contact scheme 不敏感。所以 density sector 不是 full Ward $O(q)$ 主因。

## I. spatial sector 为什么是 O(q)

spatial-spatial block $\Pi_{ij}$ 在 $q\to0$ 可以有限。若 current sign 或 Ward $q$-sign 混用，则

$$
q_i\Pi_{ij}
$$

直接留下 $O(q)$ residual。Stage 4.3C 的 spatial-current audit 显示 spatial residual 的 small-q scaling $\alpha\approx1$，与该判断一致。

## J. 下一步：formula-to-code mapping 与 targeted diagnostics

下一步首先是 Stage 4.6A formula-to-code mapping audit，也就是本节这样的对象对应审计：
明确代码里的 vertex、bubble、contact extraction 到底对应哪些解析对象。只有在这个
映射清楚后，才考虑少量 targeted diagnostic。

Hamiltonian-vector convention：

- 使用 Hamiltonian vector vertex $\Gamma_i^H$；
- Ward contraction 使用 $Q_H=(i\Omega,-q_x,-q_y)$；
- code contact extraction $C_{ij}^{\mathrm{code}}=+\langle\Lambda_{ij}^H\rangle$
  作为 diagnostic candidate。

Physical-current convention：

- 使用 physical current vertex $\Gamma_i^{\mathrm{phys}}=-\Gamma_i^H$；
- Ward contraction 使用 $Q_{\mathrm{phys}}=(i\Omega,+q_x,+q_y)$；
- physical direct contact contribution 为
  $K_{ij}^{\mathrm{phys}}=-\langle\Lambda_{ij}^H\rangle=-C_{ij}^{\mathrm{code}}$。

如果后续做 $\lambda$-scan，它只能作为 Stage 4.6B optional diagnostic
contact-coefficient scan：用来判断 residual 是否像简单 $K_{ij}^{\mathrm{phys}}$
normalization/factor
问题。$\lambda$-scan 不是确定物理系数的方法，也不能替代上述解析推导。

## 当前结论

本文档是解析符号约定推导与 formula-to-code mapping，不是最终 conductivity /
reflection / Casimir 实现。当前最可疑的是 response-level convention mismatch：
Hamiltonian vector vertex $\Gamma_i^H$、physical current vertex
$\Gamma_i^{\mathrm{phys}}$、physical direct contact contribution
$K_{ij}^{\mathrm{phys}}$、Ward vector $Q_H/Q_{\mathrm{phys}}$
与 Kubo response definition 尚未统一。`physical_current_q_plus_contact_minus`
应称为 best residual candidate / best diagnostic candidate，而不是最终 convention。
