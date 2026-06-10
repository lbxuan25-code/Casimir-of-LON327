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

## Stage 4.1B: Peierls Hamiltonian vector vertex

结果：

- Peierls Hamiltonian vector vertex $\Gamma_i^H$ 顶角级 Ward identity 通过；
- plus sign 对应 Hamiltonian derivative vertex：

$$
\Gamma_i^H(\mathbf{k},\mathbf{q})
=
+i\sum_R R_i t_R e^{i\mathbf{k}\cdot R}
\operatorname{sinc}\!\left(\frac{\mathbf{q}\cdot R}{2}\right).
$$

解释：

顶角级正确说明 $\Gamma_i^H$ 的 Hamiltonian vector vertex convention 正确，但不等于
physical current vertex $\Gamma_i^{\mathrm{phys}}=-\Gamma_i^H$ 或 final response sign 已经确定。

## Stage 4.1C: Peierls current 接入 response prototype

结果：

- 把 Peierls Hamiltonian vector vertex $\Gamma_i^H$ 接入 $\Pi_{\mu\nu}$ 后，full Ward residual 没有显著改善；
- midpoint 与 Peierls current 的 response-level residual 趋势接近。

解释：

顶角级 Ward identity 通过不保证 response-level Ward identity 闭合。问题可能在
response convention、equal-time / commutator term、physical direct contact contribution
$K_{ij}^{\mathrm{phys}}$ 或 current sign。

## Stage 4.1D / 4.2B: q0 mass 与 finite-q contact 接入

结果：

- q0 mass diagnostic contact 只带来有限改善；
- finite-q Peierls Hamiltonian contact vertex $\Lambda_{ij}^H$ 以 code contact extraction
  $C_{ij}^{\mathrm{code}}$ 接入 spatial-spatial block 后也只小幅改善；
- 二者都不能消除 small-q $O(q)$ residual；
- finite-q Peierls contact 不明显优于 q0 mass diagnostic。

解释：

finite-q contact 与 q0 mass 的差异从 $O(q^2)$ 开始，因此不能修复 leading $O(q)$ response-level 缺口。contact 的改善主要是中大 $q$ 的数值效果，不是 small-q leading 闭合。

## Stage 4.2A: Peierls Hamiltonian contact vertex audit

结果：

- finite-q Peierls Hamiltonian contact vertex $\Lambda_{ij}^H$ 的 $q=0$ mass limit 通过；
- Hermiticity 通过；
- $\Lambda_{xy}=\Lambda_{yx}$ 通过。

解释：

Hamiltonian contact vertex $\Lambda_{ij}^H$ 本身基本可信。后续问题更可能在
response-level physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ 的符号、
归一化或 Kubo convention。

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
- spatial raw residual 中 contact-minus candidate 更小；
- left/right spatial residuals close。

解释：

spatial-spatial block 在 $q\to0$ 可以有限，因此 current sign / Ward $q$-sign /
contact sign mismatch 会直接留下 $O(q)$ residual。contact-minus candidate 更小是
response-level sign convention 的线索，不是最终实现选择。

## Stage 4.4: response-level convention diagnostic

结果：

- baseline `current_code_phys_q_plus` 的 max full/spatial residual 约为 $0.122$；
- contact-minus candidates 的 max full/spatial residual 约为 $0.0415$；
- `physical_current_q_plus_contact_minus` 和
  `hamiltonian_vertex_q_minus_contact_minus` 给出相近的最小 residual；
- 但所有 case 的 spatial small-q scaling 仍为 $\alpha\approx1$，没有提升到
  $O(q^2)$；
- density residual 在一致 convention 下接近机器精度。

解释：

contact-minus candidate 与 physical direct contact contribution
$K_{ij}^{\mathrm{phys}}=-\langle\Lambda_{ij}^H\rangle=-C_{ij}^{\mathrm{code}}$ 的方向一致，
并得到 residual 诊断支持。但这只是
best residual candidate / best diagnostic candidate，不是最终 convention。
Residual minimization is not a physical derivation；residual 最小不能替代物理推导。

## Stage 4.5: best diagnostic candidate spatial term decomposition

结果：

- best diagnostic candidate 的 max residual 仍约为 $0.0415$；
- residual 仍为 $O(q)$；
- longitudinal channel 是主问题；
- transverse channel 接近机器精度；
- code contact extraction $C_{ij}^{\mathrm{code}}$ / physical direct contact contribution
  $K_{ij}^{\mathrm{phys}}$ 是 leading $O(q)$ 量级，并能降低 residual 系数；
- left/right comparison 暴露出仍需检查 response index order / conjugation。

解释：

Stage 4.5 显示 $C_{ij}^{\mathrm{code}}$ 与 $K_{ij}^{\mathrm{phys}}$ 相关的 direct
contact contribution 参与 leading $O(q)$ cancellation，但没有把 Ward
residual 闭合。当前 leftover 更像 contact-sensitive 的 response-level mismatch，
可能涉及 contact normalization/factor/sign、equal-time / commutator term、
Kubo bubble convention 或 response index order。不能据此声称 contact minus 已解决问题，
也不能声称 Ward identity 已闭合。

## 综合判断

当前证据链支持：

- Peierls Hamiltonian vector/contact vertices $\Gamma_i^H,\Lambda_{ij}^H$ 本身不是主要问题；
- density sector 不是 $O(q)$ 主因；
- spatial-current sector 是 full Ward $O(q)$ 缺口来源；
- physical direct contact contribution candidate $K_{ij}^{\mathrm{phys}}=-C_{ij}^{\mathrm{code}}$
  能改善 spatial residual，但不能闭合 small-q leading term；
- finite-q Peierls contact 不明显优于 q0 mass diagnostic，符合二者差异为 $O(q^2)$ 的解析判断；
- contact-minus candidate 更小提示 physical current vertex $\Gamma_i^{\mathrm{phys}}$
  和 physical direct contact contribution $K_{ij}^{\mathrm{phys}}$ 可能与当前 prototype
  convention 不一致；它有解析动机，但不是最终物理实现。

最可疑的问题是 response-level convention mismatch：

- Hamiltonian vector vertex $\Gamma_i^H$ 与 physical current vertex
  $\Gamma_i^{\mathrm{phys}}=-\Gamma_i^H$ 的符号；
- code contact extraction $C_{ij}^{\mathrm{code}}$ 与 physical direct contact contribution
  $K_{ij}^{\mathrm{phys}}=-C_{ij}^{\mathrm{code}}$ 的符号；
- Ward contraction 中 $Q_H=(i\Omega,-q_x,-q_y)$ 与
  $Q_{\mathrm{phys}}=(i\Omega,+q_x,+q_y)$ 的选择；
- Kubo bubble sign、denominator、complex conjugation 和 $\Pi_{\mu\nu}$ 指标顺序；
- equal-time / commutator term 与 paramagnetic-diamagnetic cancellation。

## 下一步

下一步应先做 Stage 4.6A formula-to-code mapping audit，而不是继续 residual 参数扫描：

1. 明确 `peierls_current_vertex(sign_convention="plus")` 对应 $\Gamma_i^H$；
2. 明确 `peierls_contact_vertex` 对应 $\Lambda_{ij}^H$；
3. 明确 code plus-contact extraction 中 `contact_only=+\langle\Lambda_{ij}^H\rangle`；
4. 明确 physical direct contact contribution
   $K_{ij}^{\mathrm{phys}}=-\langle\Lambda_{ij}^H\rangle=-C_{ij}^{\mathrm{code}}$；
5. 再检查 Kubo bubble sign、denominator、matrix-element order、equal-time /
   commutator term 和 response index order。

可选的 Stage 4.6B $\lambda$-scan 只能作为 diagnostic contact-coefficient scan，
用来判断 residual 是否像简单 contact normalization/factor 问题。它不是确定物理系数
的方法。在 closure 通过前，不把这些结果接入 finite-q conductivity、reflection 或
Casimir torque。
