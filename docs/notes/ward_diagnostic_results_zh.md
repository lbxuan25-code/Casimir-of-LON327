# Ward diagnostic 结果汇总

## 用途

本文档汇总 finite-q response / Ward identity 相关诊断结果，形成后续复查的证据链。它不是最终 conductivity、reflection 或 Casimir 结论，也不是材料物理结论。

相关文档：

- [pipeline 总览](casimir_torque_response_pipeline_zh.md)
- [Peierls 顶角约定](peierls_vertex_convention_zh.md)
- [response-level Ward 符号推导](response_level_ward_convention_derivation_zh.md)

## Stage 4.1A: TB Fourier reconstruction

结果：

- $H_0^{\mathrm{hop}}(\mathbf{k})=\sum_R t_R e^{i\mathbf{k}\cdot R}$ 对现有三角函数 $H_0^{\mathrm{trig}}(\mathbf{k})$ 的重构通过；
- hopping/Fourier 表示与 trigonometric 表示是同一个 Hamiltonian 的两种表达；
- hopping/Fourier 表示不是新模型，也不是更高精度模型。

解释：

该阶段建立了 Peierls finite-q 顶角的输入基础。

## Stage 4.1B: Peierls current vertex

结果：

- Peierls current vertex 顶角级 Ward identity 通过；
- plus sign 对应 Hamiltonian derivative vertex：

$$
\Gamma_i^H(\mathbf{k},\mathbf{q})
=
+i\sum_R R_i t_R e^{i\mathbf{k}\cdot R}
\operatorname{sinc}\!\left(\frac{\mathbf{q}\cdot R}{2}\right).
$$

解释：

顶角级正确说明 $\Gamma_i^H$ 的 Hamiltonian derivative convention 正确，但不等于 physical current response sign 已经确定。

## Stage 4.1C: Peierls current 接入 response prototype

结果：

- 把 Peierls current vertex 接入 $\Pi_{\mu\nu}$ 后，full Ward residual 没有显著改善；
- midpoint 与 Peierls current 的 response-level residual 趋势接近。

解释：

顶角级 Ward identity 通过不保证 response-level Ward identity 闭合。问题可能在 response convention、equal-time/contact term 或 current sign。

## Stage 4.1D / 4.2B: q0 mass 与 finite-q contact 接入

结果：

- q0 mass diagnostic contact 只带来有限改善；
- finite-q Peierls contact 接入 spatial-spatial block 后也只小幅改善；
- 二者都不能消除 small-q $O(q)$ residual；
- finite-q Peierls contact 不明显优于 q0 mass diagnostic。

解释：

finite-q contact 与 q0 mass 的差异从 $O(q^2)$ 开始，因此不能修复 leading $O(q)$ response-level 缺口。contact 的改善主要是中大 $q$ 的数值效果，不是 small-q leading 闭合。

## Stage 4.2A: Peierls contact vertex audit

结果：

- finite-q Peierls contact vertex 的 $q=0$ mass limit 通过；
- Hermiticity 通过；
- $\Lambda_{xy}=\Lambda_{yx}$ 通过。

解释：

contact 顶角本身基本可信。后续问题更可能在 response-level direct term 的符号、归一化或 Kubo convention。

## Stage 4.3A: full Ward decomposition

结果：

- full Ward residual decomposition 显示 small-q scaling $\alpha\approx1$；
- 存在稳定的 response-level $O(q)$ 缺口；
- left/right structures 接近。

解释：

这提示不是随机数值噪声，而是系统性的 response convention 或 equal-time/contact closure 问题。

## Stage 4.3B: density-current sector

结果：

- density-current sector residual 的 small-q scaling $\alpha\approx2$；
- density residual 对 contact scheme 不敏感；
- peierls+none 与 peierls+finite_q_peierls+plus 的 density residual 几乎一样；
- left/right density residual 接近。

解释：

density sector 不是 full Ward $O(q)$ 主因。$\Gamma_0=I_4$ 与 orbital embedding 仍可作为后续一致性检查，但当前 leading $O(q)$ 问题更可能不在 density-current block。

## Stage 4.3C: spatial-current sector

结果：

- spatial-current sector residual 的 small-q scaling $\alpha\approx1$；
- spatial sector 是 full Ward $O(q)$ 来源；
- contact improves but does not close；
- finite-q contact not materially better than q0 mass；
- spatial raw residual 中 contact minus sign better；
- left/right spatial residuals close。

解释：

spatial-spatial block 在 $q\to0$ 可以有限，因此 current sign / Ward $q$-sign / contact sign mismatch 会直接留下 $O(q)$ residual。contact minus sign 更好是 response-level sign convention 的线索，不是最终实现选择。

## 综合判断

当前证据链支持：

- Peierls current/contact 顶角本身不是主要问题；
- density sector 不是 $O(q)$ 主因；
- spatial-current sector 是 full Ward $O(q)$ 缺口来源；
- contact term 能改善 spatial residual，但不能闭合 small-q leading term；
- finite-q Peierls contact 不明显优于 q0 mass diagnostic，符合二者差异为 $O(q^2)$ 的解析判断；
- contact minus sign 更好提示 physical-current/contact response sign 可能与当前 prototype convention 不一致。

最可疑的问题是 response-level convention mismatch：

- Hamiltonian derivative current vertex $\Gamma_i^H$ 与 physical current $j_i=-\Gamma_i^H$ 的符号；
- contact direct term 的符号；
- Ward contraction 中 $Q_H=(i\Omega,-q_x,-q_y)$ 与 $Q_{\mathrm{phys}}=(i\Omega,+q_x,+q_y)$ 的选择；
- Kubo bubble sign、denominator、complex conjugation 和 $\Pi_{\mu\nu}$ 指标顺序；
- equal-time / commutator term 与 paramagnetic-diamagnetic cancellation。

## 下一步

下一步应做 convention verification，而不是继续盲目扫参数：

1. 比较 Hamiltonian-vertex convention 与 physical-current convention；
2. 同时切换 current sign、Ward $q$-sign、contact sign；
3. 检查 spatial current-current bubble 与 equal-time/direct term 是否满足同一 response-level Ward identity；
4. 在 closure 通过前，不把这些结果接入 finite-q conductivity、reflection 或 Casimir torque。
