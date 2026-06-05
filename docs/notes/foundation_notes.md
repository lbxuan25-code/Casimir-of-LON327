# 稳定基础与计算 Contract

本文记录当前仓库采用的稳定模型、local-response 计算 contract 和结果解释边界。
当前总状态见 `docs/reports/current_project_status.md`；local-response baseline
见 `docs/reports/local_response_baseline_status.md`；数值可信度证据见
`docs/notes/numerical_stability_summary.md`。

## 1. 仓库分层

- `src/lno327/`：稳定物理实现与公共接口。
- `scripts/`：材料本征结果和当前主计算入口。
- `outputs/`：当前主结果与边界清楚的初级结论。
- `validation/scripts/`：收敛性、静态规范、单位和 smoke 诊断。
- `validation/outputs/`：验证结果与 response diagnostic 输出。

`scripts/` 与 `outputs/` 不再保存 quick、example、compatibility wrapper 或历史
convergence 结果。

## 2. Normal-State 模型

normal-state Hamiltonian 使用四轨道基

$$
\Psi_{\mathbf{k}} =
\left(d_{z^2,1}, d_{x^2-y^2,1}, d_{z^2,2}, d_{x^2-y^2,2}\right)^T ,
$$

其 block 结构为

$$
H_0(\mathbf{k}) =
\begin{pmatrix}
H_{\parallel}(\mathbf{k}) & H_{\perp}(\mathbf{k}) \\
H_{\perp}(\mathbf{k}) & H_{\parallel}(\mathbf{k})
\end{pmatrix}
- \mu I ,
$$

当前默认化学势为

$$
\mu = 0.05\ \mathrm{eV}.
$$

本轮整理未修改 normal-state Hamiltonian 或其参数。

## 3. 最小配对 Ansatz

$s_{\pm}$ 配对采用层间 $d_{z^2}$ 结构：

$$
\Delta_{s_{\pm}}
= \Delta_0
\begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}.
$$

$d$-wave / $B_{1g}$ 配对采用同层
$d_{z^2}$-$d_{x^2-y^2}$ interorbital 结构：

$$
\Delta_d(\mathbf{k})
= \Delta_0 \left[\cos(k_x) + \cos(k_y)\right]
\begin{pmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{pmatrix}.
$$

两类 pairing 均满足

$$
\Delta(\mathbf{k}) = \Delta^T(-\mathbf{k}).
$$

本轮整理未修改 pairing ansatz。

## 4. BdG Local-Response Contract

BdG Hamiltonian 为

$$
H_{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
H_0(\mathbf{k}) & \Delta(\mathbf{k}) \\
\Delta^\dagger(\mathbf{k}) & -H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

charge-current vertex 使用

$$
J_a^{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
\partial_a H_0(\mathbf{k}) & 0 \\
0 & -\partial_a H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

mass/contact vertex 使用

$$
M_{ab}^{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
\partial_a\partial_b H_0(\mathbf{k}) & 0 \\
0 & -\partial_a\partial_b H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

当前 kernel convention 已由 normal-state Peierls/free-energy stiffness 诊断固定为：

$$
K_{\mathrm{para}}
= \text{positive current-current bubble},
$$

$$
K_{\mathrm{total}}(i\xi_n)
= K_{\mathrm{dia}} - K_{\mathrm{para}}(i\xi_n).
$$

BdG paramagnetic bubble 保留：

- $m=n$ intraband / Fermi-surface 项；
- 用于补偿 particle-hole redundancy 的整体 Nambu $1/2$ prefactor。

`K_total` 不再使用旧的加法 convention。当前主输出已经按
`K_dia - K_para` contract 重新生成。

positive Matsubara 上的 sigma-like response 定义为

$$
\Sigma_{\mathrm{SC}}(i\xi_n)
= \frac{K_{\mathrm{total}}(i\xi_n)}{\omega_{\mathrm{eV},n}},
\qquad n \ge 1.
$$

它不是实频 optical conductivity，也不应被描述为最终 optical conductivity。

## 5. 静态与 Normal-Limit 解释

当前 static diagnostic 的含义是检查 electromagnetic stiffness，而不是要求
clean normal state 的 $K_{\mathrm{total}}(0)$ 必须为零。

在 $\Delta_0=0$ normal limit 下，当前诊断要求：

- BdG $K_{\mathrm{para}}$ 与 normal $K_{\mathrm{para}}$ 对齐；
- BdG $K_{\mathrm{dia}}$ 与 normal $K_{\mathrm{dia}}$ 对齐；
- BdG 与 normal 的 $K_{\mathrm{dia}}-K_{\mathrm{para}}$ 对齐；
- $C_4$ 对称性与近零 off-diagonal response 保持稳定。

相关诊断属于 `validation/`，不作为主结果目录。

## 6. 单位与响应边界

Hamiltonian、pairing、顶点与 Matsubara 能量均以 eV 为基础单位。玻色 Matsubara
能量为

$$
\omega_{\mathrm{eV},n} = \hbar\xi_n = 2\pi n k_B T.
$$

BZ 权重满足

$$
\sum_{\mathbf{k}} w_{\mathbf{k}}=1,
$$

对应

$$
\int_{\mathrm{BZ}}\frac{d^2k}{(2\pi)^2}.
$$

response 到 reflection convention 的转换由共享单位接口管理，禁止把裸 kernel
直接解释为 conductivity，也禁止重复乘单位转换因子。

## 7. Casimir 计算边界

当前已经完成 local-response Casimir Matsubara、平行动量、角度与距离扫描。当前
主计算仍具有以下明确元数据：

```text
local_response=True
finite_momentum_resolved=False
n0_policy=skip
benchmark_only=True
preliminary_local_response_conclusion=True
not_final_casimir_conclusion=True
```

`n=0` 项尚无显式 zero-frequency reflection model，因此当前采用 `skip` policy。
finite-q Stage 1 已作为 validation 层重新引入，只验证 normal current-current
kernel $K(i\omega_n,\mathbf{q})$ 的 $\mathbf{q}\to\mathbf{0}$ same-interface
收敛。它不是 gauge/Ward-closed finite-q conductivity，也不是 Casimir 输入。

当前 Casimir distance scan 可以作为 local-response zero-torque baseline 的初级结论，
但不能解释为最终 Casimir torque 结论。

## 8. 图像与结果规范

当前主结果保存在 `outputs/`，主图统一使用 publication-oriented 样式：

- 300 dpi PNG；
- serif 字体与一致的数学排版；
- 一致的颜色、线宽、刻度和图例；
- Casimir torque 图以共享 torque tolerance 归一化。

验证性图像保留在 `validation/outputs/`，不与主结果混放。
