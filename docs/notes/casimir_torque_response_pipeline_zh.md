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
-> Peierls current/contact vertices
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

- finite-q current vertex 是否满足顶角级 Ward identity；
- finite-q contact vertex 是否与同一个 Peierls 展开一致；
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
- $x,y$ 表示 spatial current vertex；
- contact / diamagnetic term 只进入 spatial-spatial block；
- Ward identity 同时约束 density-current 和 current-current block。

Peierls substitution 给出一阶 Hamiltonian derivative vertex $\Gamma_i^H$ 和二阶 contact vertex $\Lambda_{ij}^H$。它们是构造 gauge-consistent response 的必要组成，但不等于已经得到最终 conductivity。

## 当前阶段状态

已完成：

- $H_0^{\mathrm{hop}}(\mathbf{k})=\sum_R t_R e^{i\mathbf{k}\cdot R}$ 对原三角函数 $H_0(\mathbf{k})$ 的重构审计；
- Peierls current vertex 顶角级 Ward identity；
- Peierls contact vertex 的 $q\to0$ mass limit、Hermiticity、$\Lambda_{xy}=\Lambda_{yx}$ 审计；
- normal-state $\Pi_{\mu\nu}$ Ward prototype；
- full / density / spatial Ward residual decomposition。

仍是 diagnostic：

- midpoint velocity finite-q kernel；
- Peierls current/contact 接入后的 normal-state Ward response；
- q0 mass diagnostic contact；
- finite-q Peierls contact response-level sign 比较。

尚未完成：

- response-level convention 的最终闭合；
- final finite-q conductivity；
- reflection matrix 接入；
- Casimir energy / torque 计算。

## 当前边界

所有 finite-q Ward 输出目前都只是 diagnostic。它们不是 conductivity，不是 reflection/Casimir input，也不是材料结论。下一步应优先验证 response-level convention：current sign、contact sign、Ward $q$-sign、Kubo bubble sign、equal-time/contact term 与 $\Pi_{\mu\nu}$ 指标顺序。
