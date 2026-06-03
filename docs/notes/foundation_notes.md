# 基础说明

本仓库当前实现的是底层代数结构、响应函数诊断和 Casimir 前置接口，不执行正式
Casimir Matsubara 求和，也不声明最终数值结论。当前工作重心是研究 $s_{\pm}$
与 $d$-wave 在 gap structure 和 conductivity symmetry 上的区别；Casimir
力矩是后续应用层。

输出图像和数据的使用边界见 `outputs/README.md`；面向论文草稿的图片选择、
caption 叙事和重画建议见 `docs/notes/publication_output_guide.md`。当前图像输出以
300 dpi PNG 为主，并保留对应 `.npz` / `.csv` 数据。

## Normal-State 模型

normal-state Hamiltonian 使用四轨道基

$$
\Psi_{\mathbf{k}} =
\left(d_{z^2,1}, d_{x^2-y^2,1}, d_{z^2,2}, d_{x^2-y^2,2}\right)^T .
$$

代码中采用的 block 结构可写为

$$
H_0(\mathbf{k}) =
\begin{pmatrix}
H_{\parallel}(\mathbf{k}) & H_{\perp}(\mathbf{k}) \\
H_{\perp}(\mathbf{k}) & H_{\parallel}(\mathbf{k})
\end{pmatrix}
- \mu I .
$$

其中 $H_{\parallel}$ 和 $H_{\perp}$ 由项目采用的
$T_z(\mathbf{k})$、$T_x(\mathbf{k})$、$T_z^\perp(\mathbf{k})$、
$T_x^\perp(\mathbf{k})$、$V(\mathbf{k})$ 与 $V'(\mathbf{k})$ 给出。
化学势作为 normal-state 参数保存，当前默认值为

$$
\mu = 0.05\ \mathrm{eV}.
$$

## 最小配对 Ansatz

$s_{\pm}$ 配对采用 $(d_{z^2,1}, d_{x^2-y^2,1}, d_{z^2,2}, d_{x^2-y^2,2})$
基下的层间 $d_{z^2}$ 结构：

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

它表示 bilayer bonding / antibonding sign-changing $s_{\pm}$ pairing。

$d$-wave 配对采用同层 $d_{z^2}$-$d_{x^2-y^2}$ interorbital 结构：

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

动量因子 $\cos(k_x)+\cos(k_y)$ 属于 $A_{1g}$，结合
$d_{x^2-y^2}$ 轨道自身的 $B_{1g}$ 对称性后，总配对属于 $d$-wave /
$B_{1g}$ 通道。两类配对矩阵均按偶宇称 spin-singlet 形式实现，满足

$$
\Delta(\mathbf{k}) = \Delta^T(-\mathbf{k}) .
$$

## BdG 结构

BdG Hamiltonian 使用 Nambu 基，结构为

$$
H_{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
H_0(\mathbf{k}) & \Delta(\mathbf{k}) \\
\Delta^\dagger(\mathbf{k}) & -H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

电流顶点不取 $\partial H_{\mathrm{BdG}}/\partial k_a$，而采用 charge-current
block 结构：

$$
J_a^{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
\partial_a H_0(\mathbf{k}) & 0 \\
0 & -\partial_a H_0^T(-\mathbf{k})
\end{pmatrix},
\qquad a \in \{x,y\}.
$$

diamagnetic vertex 只来自 normal-state Hamiltonian 的二阶导数，不包含
$\partial_{\mathbf{k}}\Delta$ 项：

$$
M_{ab}^{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
\partial_a\partial_b H_0(\mathbf{k}) & 0 \\
0 & -\partial_a\partial_b H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

当前的 total electromagnetic kernel 诊断定义为

$$
K_{\mathrm{total}}(i\xi_n)
= K_{\mathrm{para}}(i\xi_n) + K_{\mathrm{dia}} .
$$

虚频轴 superconducting sheet response kernel 定义为

$$
\Sigma_{\mathrm{SC}}(i\xi_n)
= \frac{K_{\mathrm{total}}(i\xi_n)}{\omega_{\mathrm{eV},n}},
\qquad n \ge 1 .
$$

$n=0$ 项由于除零问题当前不用于 $\Sigma_{\mathrm{SC}}$，在 Casimir 阶段前仍需单独处理。
当前不定义 $\Sigma_{\mathrm{SC}}(0)=K_{\mathrm{total}}(0)/0$。若输出
$K_{\mathrm{total}}(0)$，它只作为 stiffness-like 静态核诊断，不作为 sheet
conductivity，也不直接输入 reflection matrix。

$\Delta_0\rightarrow 0$ benchmark 用于检查 BdG response 层本身是否连续、有限、
稳定。关闭 pairing 时，`spm` 与 `dwave` 应回到共同的 BdG normal limit；但
normal-state Kubo $\sigma(i\xi)$ 与 BdG $\Sigma_{\mathrm{SC}}$ 的归一化和公式结构
不同，因此不要求逐项完全相等。若该 benchmark 出现发散、强不连续或明显 C4
对称性破坏，应先处理 response 层，不进入 Casimir 积分。

imaginary-axis response 收敛性 benchmark 用于检查 `nk`、`eta` 与 Matsubara index
对 normal / `spm` / `dwave` local response 的影响。该检查只处理 local
$q=0$ response，不包含有限动量 response。若 response 未随 `nk`
增大稳定，或随 `eta` 减小出现异常发散 / 随机震荡，应先解决数值收敛问题；若
`spm` / `dwave` 差异只在小 `nk` 或特定 `eta` 下出现，应视为数值伪影，不进入
Casimir 积分。

高 `Nk` 聚焦复查用于确认 normal response 的低 Matsubara 频率敏感性是否能在
更细网格下缓解。若 normal response 在 `Nk=64\rightarrow 80` 仍未稳定，说明当前
local response 仍需要更细采样或 Fermi-surface sensitive integration。若
`spm` / `dwave` 差异在高 `Nk` 下趋近 0，则当前 minimal pairing 的 local response
差异不应作为稳健物理差异解读；若差异在高 `Nk` 下平台化，才进入后续物理分析。

normal-state low-Matsubara sampling convergence 诊断用于进一步定位上述 normal
response 的 `Nk` 敏感来源。该诊断复用现有 Kubo 公式；`shifted` 和 `average`
mesh 只作为脚本内部的 alternative sampling，不改变默认 uniform 行为。若
points-within-eta 太少、网格命中费米面不规则，或 response 与 Fermi-window weight
强相关，应优先改进 normal response 的采样 / 积分，而不是进入 Casimir 积分。

normal-state FS-sensitive sampling benchmark 是上述诊断的下一步。`multishift_average`
使用 `s x s` fractional shifted meshes 对同一 Kubo integrand 做采样平均，并报告
shift-to-shift std；`fs_window_refined` 先用 coarse mesh 找到 Fermi-window cells，再在
这些 cell 周围局部加密并保留面积权重。二者都只改变 quadrature，不改变 normal Kubo
物理公式，也不替代 uniform 默认。若这些方案仍无法稳定 low-Matsubara normal response，
下一步应考虑 contour 或 tetrahedron Fermi-surface integration；在此之前仍不进入正式
local-response Casimir 积分。

normal-state FS-adaptive BZ integration prototype 是更稳妥的局部加密版本：它用
coarse cell 顶点和中心能量识别费米面穿过的 BZ cells，或识别落入
`fs_window_factor * max(eta,kBT,omega_eV)` 的近费米窗口 cells，然后只细分这些 cells。
非 FS cells 保留 midpoint；所有采样点按面积权重归一后送入原
`kubo_conductivity_imag_axis`。该 prototype 不做 strict contour / triangle 解析积分，
也不改变 Kubo integrand。若它仍不随 `refine_factor` 和 `Nk` 稳定，下一步应转向
triangle / contour Fermi-surface integration；normal response 收敛前仍暂停正式
local-response Casimir 积分。

## 单位与归一化

所有 Hamiltonian、配对、速度顶点和 Kubo 响应中的能量单位均为 eV。速度算符定义为

$$
v_a(\mathbf{k}) = \partial_a H_0(\mathbf{k}),
\qquad a \in \{x,y\},
$$

其中 $k_x,k_y$ 是无量纲晶格动量，因此 $v_a$ 在代码中按 eV 处理。
玻色 Matsubara 能量写作

$$
\hbar \xi_n = 2\pi n k_B T ,
$$

并以 eV 形式传入响应函数。normal-state Kubo 可选择乘以 $e^2/\hbar$，
但当前 local sheet response interface 仍标记为 model-units diagnostic；
SI sheet conductivity 归一化尚未最终完成。

BZ 积分规范为

$$
\sum_{\mathbf{k}} w_{\mathbf{k}}, \qquad \sum_{\mathbf{k}} w_{\mathbf{k}} = 1,
$$

对应在 $[-\pi,\pi)^2$ 上的

$$
\int_{\mathrm{BZ}} \frac{d^2 k}{(2\pi)^2}.
$$

## Casimir 接口边界

Casimir 相关工具当前只保留流程骨架：

1. 准备虚频轴二维响应张量；
2. 按板间相对角度或面内角度旋转张量；
3. 构造 reflection matrix；
4. 构造 Lifshitz 能量 integrand；
5. 由 $-\partial_\theta E$ 构造力矩 integrand。

Lifshitz 求和形式上包含 $n=0$ Matsubara 半权重项。当前 local isotropic baseline
默认 `n=0 policy = skip`，不是因为 $n=0$ 项不存在，而是为了避免当前未定义的
superconducting zero-frequency conductivity 产生假贡献。`extrapolate_from_lowest_matsubara`
只作为数值敏感性估计；`use_static_kernel` 只作为静态核诊断。
`skip` 不是无条件物理结论：当前仅在 extrapolated $n=0$ proxy 相对于
$n\ge 1$ torque-integrand partial sum 的影响低于阈值，或两者均为数值零时，
才接受为 local baseline 的保守默认。若 proxy 影响超过阈值，或 reference torque
近零但 proxy 非零，则必须先建立 zero-frequency reflection model，不能输出正式
Casimir torque 结论。

当前新增的 `LocalSheetResponse` 只是 Casimir 前置接口，把 normal-state
$\sigma(i\xi)$ 与 BdG $\Sigma_{\mathrm{SC}}(i\xi)$ 统一整理为 local $q=0$
sheet response matrix。该对象当前保持
`valid_for_casimir_input=False`，因为正式 Casimir 阶段仍缺少
$n=0$ Matsubara 处理、SI sheet conductivity 归一化、非局域
$q_{\parallel}$ 响应，以及能产生 torque 的角向各向异性机制。
未来若引入真实各向异性机制，必须重新推导或选择相应的 $n=0$ zero-frequency
reflection policy，然后才可进入正式 Matsubara 求和。
当前 n0 sensitivity 判断仍固定 $k_{\parallel},\phi,\theta$，并只做 partial
Matsubara-sum integrand 比较；它不是完整 $k_{\parallel}/\phi$ 积分后的最终判断。
