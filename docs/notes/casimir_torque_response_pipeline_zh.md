# Casimir torque response pipeline 总览

## 用途

本文档记录本项目从 normal-state Hamiltonian 到未来 finite-q Casimir torque 输入的计算链条。
它是路线图和边界说明，不是最终 Casimir torque 结论，不是材料物理结论。

相关细节见：

- [Peierls 顶角约定](peierls_vertex_convention_zh.md)
- [response-level Ward 符号推导](response_level_ward_convention_derivation_zh.md)
- [Ward 诊断结果汇总](ward_diagnostic_results_zh.md)
- [finite-q 电磁耦合约定](finite_q_electromagnetic_coupling_convention_zh.md)

## 总流程

当前目标不是直接计算 Casimir torque，而是先建立未来 finite-q response 的 gauge-consistent 基础。完整链条应为：

```text
H0(k)
-> tight-binding Fourier/hopping representation
-> Peierls Hamiltonian vector/contact vertices (V_i, M_ij)
-> Pi_{mu nu}(iOmega, q)
-> Ward identity
-> future finite-q conductivity
-> future reflection matrix
-> future Casimir energy/torque
```

其中每一步都必须保留可追溯的符号约定和 provenance 字段。当前已经做到的是 Hamiltonian 表示、Peierls 顶角、contact 顶角和 normal-state Ward diagnostic；尚未得到 final finite-q conductivity、reflection matrix 或 Casimir torque。

## 为什么 local q=0 conductivity 不足够

Casimir 几何中的电磁涨落带有 in-plane momentum $\mathbf{q}$。local $q=0$ conductivity 只描述空间均匀外场极限，不能代表有限横向动量下的 density/current response。对于未来 reflection matrix 和 Casimir energy，需要的是与外场 momentum $\mathbf{q}$ 一致的 finite-q response，而不是把 local conductivity 简单外推。

因此，当前阶段必须先检查：

- finite-q Hamiltonian vector vertex $V_i$ 是否满足顶角级 Ward identity；
- finite-q Hamiltonian contact vertex $M_{ij}$ 是否与同一个 Peierls 展开一致；
- $\Pi_{\mu\nu}(i\Omega,\mathbf{q})$ 是否在 response level gauge consistent；
- density/current/contact 的符号、频率、指标顺序是否自洽。

## q 与 iOmega 的物理意义

$\mathbf{q}$ 是外部电磁场或真空涨落带入材料表面的 in-plane momentum。它不是电子内部求和变量 $\mathbf{k}$。实际 bubble 计算中，电子态使用 shifted momenta $\mathbf{k}\pm\mathbf{q}/2$，而 $\mathbf{q}$ 是外部 probe。

$i\Omega_n$ 是 bosonic Matsubara frequency，属于外部电磁扰动频率。它不是 band eigenvalue，也不是费米能附近的单粒子能级。

## density / current / contact 的位置

response prototype 记为

$$
\Pi_{\mu\nu}(i\Omega_n,\mathbf{q}),\qquad \mu,\nu=0,x,y .
$$

其中：

- $0$ 表示 density vertex，目前 prototype 使用 $\Gamma_0=I_4$；
- $x,y$ 表示 spatial current/current-like vertex，需要区分 Hamiltonian vector vertex
  $V_i$ 和 physical current vertex $j_i=-V_i$；
- physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ 只进入 spatial-spatial block；
- Ward identity 同时约束 density-current 和 current-current block。

Peierls substitution 给出一阶 Hamiltonian vector vertex $V_i$ 和二阶
Hamiltonian contact vertex $M_{ij}$。physical-current response 还需要区分
code contact extraction $C_{ij}^{\mathrm{code}}=+\langle M_{ij}\rangle$ 与
physical direct contact contribution
$K_{ij}^{\mathrm{phys}}=-C_{ij}^{\mathrm{code}}$。这些对象是构造 gauge-consistent
response 的必要组成，但不等于已经得到最终 conductivity。

## 当前阶段状态

当前活跃阶段是 Stage 4.7 API cleanup 之后的 response-level Ward convention verification，
而不是早期 Stage 1 / Stage 2
规划，也不是继续做 residual 参数扫描。Stage 1--3 的结果仍保留为 diagnostic evidence；它们不能替代完整
$\Pi_{\mu\nu}$ Ward closure。

已完成：

- $H_0^{\mathrm{hop}}(\mathbf{k})=\sum_R t_R e^{i\mathbf{k}\cdot R}$ 对原三角函数 $H_0(\mathbf{k})$ 的重构审计；
- Peierls Hamiltonian vector vertex $V_i$ 顶角级 Ward identity；
- Peierls Hamiltonian contact vertex $M_{ij}$ 的 $q\to0$ mass limit、Hermiticity、$M_{xy}=M_{yx}$ 审计；
- normal-state $\Pi_{\mu\nu}$ Ward prototype；
- full / density / spatial Ward residual decomposition；
- Stage 4.4 / 4.5 response-level convention diagnostic 与 spatial term decomposition；
- Stage 4.7 destructive API cleanup：主代码对象命名为 $V_i$、$M_{ij}$、$j_i=-V_i$，不再用
  `sign_convention="plus"` 构造 $V_i$。

仍是 diagnostic：

- midpoint velocity finite-q kernel；
- Peierls Hamiltonian vector/contact vertices $V_i,M_{ij}$ 接入后的 normal-state Ward response；
- q0 mass diagnostic contact；
- finite-q Peierls contact response-level sign 比较；
- best residual candidate / best diagnostic candidate 的 term decomposition。

尚未完成：

- response-level convention 的最终闭合；
- formula-to-code mapping 审计后的 Kubo bubble / equal-time / index-order 复查；
- final finite-q conductivity；
- reflection matrix 接入；
- Casimir energy / torque 计算。

## Stage roadmap / 远期阶段

### Stage 2 / 2.5: Casimir q-grid 到 model-q 的单位审计

Casimir 积分中的无量纲变量 $u$ 和层间距离 $d$ 映射到模型动量：

$$
q_{\mathrm{model}}=a_{\parallel}\frac{u}{d}.
$$

这里 $a_{\parallel}$ 是面内赝四方 / Ni-Ni 有效晶格常数，用于从 SI 动量换算到
model-q；$d$ 是层间距离，必须与 $a_{\parallel}$ 使用同一长度单位。

Stage 2 / 2.5 只是单位和采样范围审计：

- 不计算 response；
- 不产生 finite-q conductivity；
- 不接入 reflection 或 Casimir integral；
- 不声明 Casimir 结论。

$a_{\parallel}$ sensitivity 可作为 q-grid 采样范围的系统误差检查，用来判断
Casimir-relevant $q_{\mathrm{model}}$ 是否落在 Stage 1 小 $q$ diagnostic 覆盖范围内。

### Stage 3: BdG finite-q current-current diagnostic

Stage 3 的对象仍是 current-current kernel diagnostic。它需要检查两个极限：

$$
\Delta_0\to0
\Rightarrow
\text{normal finite-q kernel limit},
$$

以及

$$
\mathbf{q}\to\mathbf{0}
\Rightarrow
\text{local BdG kernel limit}.
$$

这不是 gauge-closed finite-q conductivity。未来若进入 BdG gauge-consistent response，
需要额外考虑 collective phase / vertex correction；bare current-current block 不能直接作为
reflection/Casimir input。

### Stage 4.6A / 4.7: 当前 response-level 收尾路线

Stage 4.6A formula-to-code mapping audit 已把代码对象与解析对象明确对应。Stage 4.7
把主 API 清理为固定对象：

```text
peierls_hamiltonian_vector_vertex = V_i
physical current vertex = j_i = -V_i
peierls_hamiltonian_contact_vertex = M_ij
code contact extraction C_ij^code = contact_only = +<M_ij>
physical direct contact contribution K_ij^phys = -C_ij^code
Pi_ij^candidate = bubble[V_i,V_j] - <M_ij>
```

这一步的目标是防止把 residual 最小的组合误读为最终物理实现。
`physical_current_q_plus_contact_minus` 只能称为 best residual candidate /
best diagnostic candidate：它有 physical direct contact contribution
$K_{ij}^{\mathrm{phys}}$ 的解析动机，并降低
residual，但 Ward residual 仍为 $O(q)$，不能声明闭合。

Stage 4.6B 可作为 optional diagnostic contact-coefficient scan。$\lambda$-scan 必须在
上述标准对象命名稳定后才有意义；它不是确定物理系数的方法，不能把 $\lambda$ 当作自由
物理参数拟合，只能用来判断 residual 是否像简单 $K_{ij}^{\mathrm{phys}}$
normalization/factor 问题。
在 formula-to-code mapping、Kubo bubble sign、equal-time / commutator term、denominator、
matrix-element order 和 response index order 未闭合前，不进入 finite-q conductivity、
reflection 或 Casimir。

### Stage 5: future reflection / Casimir benchmark 接入

只有在 Stage 4 response-level Ward convention 闭合后，才允许把 finite-q response
接入 reflection/Casimir benchmark。closure 通过前，不应把 current-current-only diagnostic
接入 Casimir。

未来 response cache 至少需要记录：

```text
qx_model
qy_model
matsubara_n
frequency_mode
temperature_K
response_kind
unit_convention
```

这些字段用于追踪外部 momentum、frequency convention、温度、response 类型和单位约定。
这一阶段仍应先作为 benchmark，不直接声明最终材料结论。

### Stage 6: 各向异性机制 benchmark

在 finite-q response / reflection / benchmark 都稳定后，才讨论 anisotropy mechanism：

- finite-q crystal harmonic；
- pairing symmetry；
- normal-state anisotropy；
- superconducting kernel correction；
- torque-like observables 的相对贡献比较。

这些方向只能作为远期机制 benchmark。不能在 finite-q response 和 reflection/Casimir
benchmark 稳定前提前声明材料结论。

## 当前边界

所有 finite-q Ward 输出目前都只是 diagnostic。它们不是 conductivity，不是
reflection/Casimir input，也不是材料结论。Residual minimization is not a physical
derivation；residual 最小不能替代物理推导。下一步应复查 current sign、contact sign、Ward $q$-sign、
Kubo bubble sign、equal-time / commutator term、physical direct contact contribution
$K_{ij}^{\mathrm{phys}}$ 与 $\Pi_{\mu\nu}$ 指标顺序。
