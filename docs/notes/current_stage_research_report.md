# 当前阶段研究结果分析汇报

## 摘要

当前仓库已经形成一条结构清楚的 local-response 主线：

$$
H_0,\ \Delta
\rightarrow H_{\mathrm{BdG}}
\rightarrow K_{\mathrm{para}},K_{\mathrm{dia}}
\rightarrow K_{\mathrm{total}}=K_{\mathrm{dia}}-K_{\mathrm{para}}
\rightarrow \Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega_{\mathrm{eV}}
\rightarrow R
\rightarrow F(d,\theta)
\rightarrow \tau=-\partial_\theta F .
$$

normal-state Hamiltonian 与 `spm` / `dwave` pairing ansatz 均未修改。当前 BdG
response 使用 positive current-current bubble、$m=n$ intraband 项和 Nambu
$1/2$ prefactor；`K_total` 已按 Peierls/free-energy stiffness 诊断更新为
`K_dia - K_para`。

当前已经完成 local-response Casimir full distance scan，并将其提升为边界清楚的
初级结论。它不是闭合测试，也不是数值稳定性输出；主结果位于：

```text
outputs/casimir/local_response_distance_scan/
```

该结论仍不是最终 Casimir torque 结论，因为 `n0_policy=skip` 且不包含有限动量响应。

## 1. 当前仓库状态

当前主计算入口按物理对象放置：

- `scripts/normal_state/`：normal-state inspection 与 conductivity。
- `scripts/pairing/`：pairing 与 gap structure。
- `scripts/bdg/`：BdG paramagnetic、diamagnetic、total kernel 与 sigma-like response。
- `scripts/casimir/`：local-response Casimir integral 与 distance scan。

当前主输出按结果对象放置：

- `outputs/normal_state/`
- `outputs/pairing/`
- `outputs/bdg/`
- `outputs/casimir/local_response_distance_scan/`

收敛性、静态规范、normal-limit、单位和 smoke 诊断只保存在
`validation/scripts/` 与 `validation/outputs/`。顶层重复 wrapper、compatibility
wrapper、example 输出和 progress-test 临时结果已经移除。

## 2. BdG Response Contract

BdG Hamiltonian 为

$$
H_{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
H_0(\mathbf{k}) & \Delta(\mathbf{k}) \\
\Delta^\dagger(\mathbf{k}) & -H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

当前 paramagnetic kernel 是正的 current-current bubble，并包含：

1. $m=n$ intraband / Fermi-surface 项；
2. 对整个 BdG bubble 统一施加的 Nambu $1/2$ prefactor。

当前 electromagnetic stiffness kernel 为

$$
K_{\mathrm{total}}(i\xi_n)
= K_{\mathrm{dia}} - K_{\mathrm{para}}(i\xi_n).
$$

positive Matsubara 上定义

$$
\Sigma_{\mathrm{SC}}(i\xi_n)
= \frac{K_{\mathrm{total}}(i\xi_n)}{\omega_{\mathrm{eV},n}},
\qquad n\ge 1.
$$

当前主 BdG 输出已经重新生成并通过 contract 审计：

```text
K_total - (K_dia - K_para) = 0
```

到保存数据的数值精度。$\Sigma_{\mathrm{SC}}$ 仍是 sigma-like response，不是最终
实频 optical conductivity。

## 3. 对称性与静态诊断结论

当前无外加各向异性机制的 normal、`spm` 与 `dwave` local response 均保持：

$$
R_{xx}\approx R_{yy},\qquad R_{xy}\approx R_{yx}\approx 0.
$$

这说明当前 local $q=0$ 主线没有产生可分辨的面内 $C_4$ 破缺。静态诊断现在解释为
stiffness diagnostic，不再错误要求 clean normal-state stiffness 必须趋近零。

normal-limit decomposition 与 Peierls/free-energy stiffness 诊断用于支持：

- BdG paramagnetic Nambu $1/2$ prefactor；
- `K_total = K_dia - K_para` convention；
- BdG 与 normal kernel 在 $\Delta_0=0$ 下的对照解释。

这些诊断属于可信度证据，保存在 `validation/outputs/response/`，不属于当前主结论。

## 4. 数值稳定性状态

当前 local-response 初级结论采用的推荐参数为：

```text
normal_sampling=fs_adaptive
normal_nk=96
normal_refine_factor=8
bdg_nk=32
phi_num=32
u_max=80
du=0.5
matsubara_max=64
```

当前状态：

- normal FS-adaptive sampling 已完成稳定性复查；
- refined cutoff scan 已达到 clean convergence；
- Matsubara normal 为 `candidate_converged`；
- Matsubara `spm` / `dwave` 为 `loose_converged`；
- 当前 local isotropic 计算保持 zero-torque baseline；
- 未发现超过共享 torque tolerance 的 spurious torque。

完整数值稳定性总结见 `numerical_stability_summary.md`。

## 5. Local-Response Casimir 初级结论

当前 full distance scan 覆盖：

```text
kinds=normal,spm,dwave
distance_m=3e-08,5e-08,7.5e-08,1e-07,1.5e-07,2e-07
theta_count=5
toy_anisotropic_control=True
```

数据与图像位于：

```text
outputs/casimir/local_response_distance_scan/
```

结果显示，normal、`spm` 与 `dwave` 在当前 local isotropic 设置下均保持
zero-torque baseline。toy anisotropic control 仍能产生非零 torque，用于确认角向
响应与积分链路具备灵敏度。

当前结论应表述为：

> 在当前 local $q=0$、`n0_policy=skip`、无显式面内各向异性机制的设置下，
> normal、minimal $s_{\pm}$ 与 minimal $d$-wave distance scan 均未显示超过数值
> tolerance 的 torque；该结果是 local-response zero-torque baseline 的初级结论。

不得把它表述为“物理 Casimir torque 必然为零”或最终 Casimir torque 结论。

## 6. 当前结果边界

distance scan 的状态字段固定为：

```text
local_response=True
finite_momentum_resolved=False
n0_policy=skip
benchmark_only=True
preliminary_local_response_conclusion=True
not_final_casimir_conclusion=True
```

主要边界为：

1. $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega_{\mathrm{eV}}$ 只定义于 $n\ge1$。
2. $n=0$ 尚无显式 zero-frequency reflection model。
3. 当前 response 仅为 local $q=0$。
4. 当前模型没有引入真实面内各向异性 torque 来源。
5. 有限动量 response prototype 已移除，不能作为当前 Casimir 输入。

## 7. 图像与输出状态

当前 `outputs/` 中主图已经统一为 publication-oriented 格式：

- 300 dpi；
- serif 字体与统一数学排版；
- 一致的颜色、线宽和刻度；
- 避免图例遮挡；
- Casimir torque 图以共享 torque tolerance 归一化。

主输出只保留当前数据与初级结论。历史、quick、收敛和诊断输出保存在
`validation/outputs/`。

## 8. 下一步

下一阶段优先级为：

1. 建立明确的 $n=0$ zero-frequency reflection policy；
2. 研究能产生可观测角向响应的真实各向异性机制；
3. 继续检验 `spm` / `dwave` local response 差异的物理稳健性；
4. 若未来重启有限动量 response，重新设计完整 response contract 与验证链；
5. 在上述缺口解决前，继续保留 `not_final_casimir_conclusion=True`。

## 9. 验证状态

本轮结构与主结果整理后：

```text
pytest -m unit: 79 passed
focused response/Casimir tests: 28 passed
pytest -m "not benchmark": 94 passed
```

这些测试支持当前工程结构和 local-response contract，但不替代最终物理模型验证。
