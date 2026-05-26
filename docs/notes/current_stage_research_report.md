# 当前阶段研究结果分析汇报

## 摘要

当前仓库已经从 pairing ansatz、BdG 基础结构、gap structure 诊断、BdG
electromagnetic kernel，到 Casimir 前置 local sheet response 接口建立了一个连贯的最小研究流程。
这一阶段的核心结论是：在无磁场、当前四轨道模型与均匀 Brillouin-zone 网格下，
normal-state response、minimal $s_{\pm}$ pairing 和 minimal $d$-wave pairing 的
local $q=0$ 虚频轴响应均保持数值上的 $C_4$ 对称性，即

$$
R_{xx}(i\xi_n) \approx R_{yy}(i\xi_n), \qquad
R_{xy}(i\xi_n) \approx R_{yx}(i\xi_n) \approx 0 .
$$

这里 $R$ 代表当前阶段统一接口中的 local sheet response matrix。对于 normal-state，
$R=\sigma(i\xi)$；对于 BdG superconducting state，
$R=\Sigma_{\mathrm{SC}}(i\xi)$。当前 $R$ 仍是 Casimir 前置诊断，不是最终
SI sheet conductivity，也不是正式 Casimir 输入。

## 模型与配对结构

四轨道基采用

$$
\Psi_{\mathbf{k}} =
\left(d_{z^2,1}, d_{x^2-y^2,1}, d_{z^2,2}, d_{x^2-y^2,2}\right)^T .
$$

normal-state Hamiltonian 写作

$$
H_0(\mathbf{k}) =
\begin{pmatrix}
H_{\parallel}(\mathbf{k}) & H_{\perp}(\mathbf{k}) \\
H_{\perp}(\mathbf{k}) & H_{\parallel}(\mathbf{k})
\end{pmatrix}
- \mu I ,
$$

其中当前默认化学势为

$$
\mu = 0.05\ \mathrm{eV}.
$$

minimal $s_{\pm}$ pairing 定义为层间 $d_{z^2}$ 配对：

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

minimal $d$-wave / $B_{1g}$ pairing 定义为同层
$d_{z^2}$-$d_{x^2-y^2}$ interorbital pairing：

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

其中 $\cos(k_x)+\cos(k_y)$ 属于 $A_{1g}$，与
$d_{x^2-y^2}$ 轨道自身的 $B_{1g}$ 对称性结合后，总体进入 $B_{1g}$ /
$d$-wave 通道。两种 pairing 均满足

$$
\Delta(\mathbf{k}) = \Delta^T(-\mathbf{k}) ,
$$

并且 pairing matrix 返回 complex dtype，便于后续处理复数相位与响应函数。

## BdG 电磁响应层次

BdG Hamiltonian 采用

$$
H_{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
H_0(\mathbf{k}) & \Delta(\mathbf{k}) \\
\Delta^\dagger(\mathbf{k}) & -H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

当前电流顶点明确采用 charge-current vertex，而不是
$\partial H_{\mathrm{BdG}}/\partial k_a$：

$$
J_a^{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
\partial_a H_0(\mathbf{k}) & 0 \\
0 & -\partial_a H_0^T(-\mathbf{k})
\end{pmatrix},
\qquad a \in \{x,y\}.
$$

diamagnetic vertex 只来自 normal-state Hamiltonian 的解析二阶导数：

$$
M_{ab}^{\mathrm{BdG}}(\mathbf{k}) =
\begin{pmatrix}
\partial_a\partial_b H_0(\mathbf{k}) & 0 \\
0 & -\partial_a\partial_b H_0^T(-\mathbf{k})
\end{pmatrix}.
$$

目前响应层次为

$$
K_{\mathrm{total}}(i\xi_n)
= K_{\mathrm{para}}(i\xi_n) + K_{\mathrm{dia}},
$$

以及

$$
\Sigma_{\mathrm{SC}}(i\xi_n)
= \frac{K_{\mathrm{total}}(i\xi_n)}{\omega_{\mathrm{eV},n}},
\qquad \omega_{\mathrm{eV},n}=\hbar\xi_n=2\pi n k_B T,\qquad n\ge 1.
$$

$n=0$ Matsubara 项当前不进入 $\Sigma_{\mathrm{SC}}$，因为定义中存在
$1/\omega_{\mathrm{eV},n}$。这不是数值细节，而是进入 Casimir 前必须解决的物理接口问题。
Lifshitz 求和形式上包含 $n=0$ 半权重项；当前仓库的 local isotropic baseline
默认 `n=0 policy = skip`，目的只是避免把未定义的 superconducting
zero-frequency conductivity 送入 reflection matrix。`extrapolate_from_lowest_matsubara`
只用于数值敏感性估计，`use_static_kernel` 只输出 stiffness-like
$K_{\mathrm{total}}(0)$ 静态核诊断。当前不定义
$\Sigma_{\mathrm{SC}}(0)=K_{\mathrm{total}}(0)/0$，也不把
$K_{\mathrm{total}}(0)$ 直接当作 sheet conductivity。

## Gap Structure 诊断结果

gap analysis 工具已经支持在 normal-state band basis 上计算投影 pairing：

$$
\Delta_n(\mathbf{k})
= u_n^\dagger(\mathbf{k})\,\Delta(\mathbf{k})\,u_n^*(-\mathbf{k}) .
$$

当前脚本可以在近似 Fermi surface 点上输出
$|\Delta_n(\mathbf{k})|$、preliminary sign、near-node count、relative node fraction
以及 band-resolved summary。需要强调的是，gap sign 当前仍是 gauge-dependent preliminary
诊断；更可靠的判断依据是

$$
|\Delta_n(\mathbf{k})|,
\qquad
\min_{\mathrm{FS}}|\Delta_n|,
\qquad
\mathrm{node\ fraction}
= \frac{N_{\mathrm{node}}}{N_{\mathrm{FS}}}.
$$

对于 minimal $d$-wave pairing，形式因子

$$
f_d(\mathbf{k})=\cos(k_x)+\cos(k_y)
$$

在 $f_d(\mathbf{k})\approx 0$ 附近天然给出 near-node 倾向。当前测试覆盖了该趋势，
但正式物理判断仍需要更细的 Fermi-surface sampling、band-resolved 分析和 tolerance
稳定性扫描。

## Local Sheet Response 对称性结果

为 Casimir 阶段前的统一接口，当前定义了 local $q=0$ response matrix：

$$
R_{\alpha\beta}(i\xi_n), \qquad \alpha,\beta\in\{x,y\}.
$$

对称性诊断使用

$$
\delta_R
= \frac{R_{xx}-R_{yy}}{R_{xx}+R_{yy}},
$$

$$
\mathrm{relative\ offdiag}
=
\frac{\sqrt{|R_{xy}|^2+|R_{yx}|^2}}
{\frac{1}{2}\left(|R_{xx}|+|R_{yy}|\right)},
$$

以及 response matrix 的 relative eigenvalue splitting。

最近一次轻量示例扫描参数为

$$
N_k = 8,\qquad
T = 30\ \mathrm{K},\qquad
\Delta_0 = 0.04\ \mathrm{eV},\qquad
n=1,2,3 .
$$

得到的对称性摘要如下：

| kind | $\max|\delta_R|$ | max relative offdiag | max relative eigen split | $|R_{xx}|$ range |
| --- | ---: | ---: | ---: | ---: |
| normal | $5.56\times 10^{-16}$ | $3.91\times 10^{-17}$ | $1.11\times 10^{-15}$ | $[0.1747, 0.1775]$ |
| $s_{\pm}$ | $5.56\times 10^{-16}$ | $6.16\times 10^{-17}$ | $1.11\times 10^{-15}$ | $[19.1629, 57.5306]$ |
| $d$-wave | $5.53\times 10^{-16}$ | $8.92\times 10^{-18}$ | $1.11\times 10^{-15}$ | $[19.2717, 57.8585]$ |

这些数值说明，在当前无磁场模型和对称网格下，normal / $s_{\pm}$ / $d$-wave
三类 local response 都没有产生可分辨的面内 $C_4$ 破缺或 off-diagonal 响应。
这与物理预期一致：如果 Hamiltonian、pairing ansatz 和数值网格都保持 $C_4$，则
local $q=0$ response 不应凭空产生 $x/y$ 各向异性。

## 对 Casimir 力矩目标的含义

Casimir 力矩需要角向各向异性。若两个板的 response 在面内完全各向同性，则仅靠旋转角
$\theta$ 不会产生有意义的 torque signal。当前结果说明，minimal $s_{\pm}$ 与
minimal $d$-wave 在 local $q=0$、无磁场、当前模型参数下都保持

$$
R(i\xi_n) \propto I_{2\times 2}
$$

到数值精度。因此，未来若要把 Casimir torque 作为区分 superconducting symmetry 的方法，
需要进一步寻找或引入能在 electromagnetic response 中体现角向结构的机制，例如：

1. 更完整的 superconducting response 归一化和 $n=0$ 处理；
2. 非局域 $q_{\parallel}$ response；
3. 更细的 Fermi-surface / band-resolved gap anisotropy 对 response 的影响；
4. 可控的晶格、轨道、应变、界面或取向机制，使 $B_{1g}$ pairing 的内部结构进入可观测的面内响应；
5. 与 reflection matrix 和 Lifshitz integrand 中角向变量的严格单位匹配。

若未来引入真实各向异性机制，$n=0$ zero-frequency reflection policy 必须重新推导
或明确选择。当前的 skip / extrapolate / static-kernel comparison 只是把这个边界
显式化，不解决最终 $n=0$ 物理问题。

## Casimir 接口链路冒烟测试

当前新增 smoke pipeline 已经打通

$$
\mathrm{LocalSheetResponse}
\rightarrow \sigma_{\alpha\beta}
\rightarrow r
\rightarrow \mathcal{E}_{\mathrm{integrand}}
\rightarrow \tau_{\mathrm{integrand}} .
$$

该链路使用 `local_response_imag_axis` 取得 normal / $s_{\pm}$ / $d$-wave 的 local
$q=0$ response，经 `conductivity_tensor_from_matrix` 转成 `ConductivityTensor`，
再调用已有 reflection matrix、能量 integrand 和力矩 integrand。该冒烟测试
的预期判据是：

$$
\tau_{\mathrm{iso}} \approx 0,
\qquad
\tau_{\mathrm{toy\ anisotropic}} \ne 0 .
$$

其中 toy anisotropic tensor 人为设置

$$
\sigma_{xx} \ne \sigma_{yy},
\qquad
\sigma_{xy}=\sigma_{yx}=0 .
$$

因此，该步骤验证的是工程链路和角向各向异性灵敏度，不是正式 Casimir 积分。
normal / $s_{\pm}$ / $d$-wave 当前 local response 可接入 Casimir integrand，但仍只作为
冒烟测试使用。

## 当前限制

当前阶段已把三个 Casimir 前置缺口转化为显式接口状态：

1. 单位归一化由 `SheetConductivityConvention` 和 `model_response_to_sheet_conductivity`
   管理，采用中性命名的三层路径：

   $$
   \sigma_{\mathrm{model}}
   \rightarrow
   \sigma_{\mathrm{sheet}}^{\mathrm{SI}}
   = \frac{e^2}{\hbar}\sigma_{\mathrm{model}}
   \rightarrow
   \sigma_{\mathrm{reflection}}
   = \frac{\sigma_{\mathrm{sheet}}^{\mathrm{SI}}}{\sigma_0}.
   $$

   当前不会把裸 model response 直接送入 reflection matrix，也会防止重复乘
   $e^2/\hbar$。
2. $n=0$ Matsubara 项由 `StaticResponsePolicy` 管理。对于 $s_{\pm}$ 与
   $d$-wave，$n=0$ 不允许直接使用
   $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega_{\mathrm{eV}}$。当前 local
   baseline 的默认选择是 skip；lowest-Matsubara extrapolation 只作为敏感性估计；
   static-kernel diagnostic 不进入 reflection matrix，也不作为 sheet conductivity。
3. nonlocal response 由 `NonlocalSheetResponse` 管理。当前实现的
   `local_fallback` 明确标记 `nonlocal_resolved=False`；真实 finite-$q_{\parallel}$
   response 仍是 `NotImplementedError` 显式占位。

当前阶段仍有以下明确限制：

1. $\Sigma_{\mathrm{SC}}(i\xi_n)$ 只定义在 $n\ge 1$。
2. Lifshitz 求和形式上包含 $n=0$ 半权重，但当前不做正式 Casimir Matsubara 求和。
3. `LocalSheetResponse.valid_for_casimir_input=False` 是有意的保守标记。
4. 当前 response 仍允许 local $q=0$ 局域回退，不包含真实非局域 $q_{\parallel}$ 计算。
5. 中性 sheet-conductivity convention 已建立，但它还需要与 $n=0$ 和 finite-$q$
   物理方案共同组成正式 Casimir 输入。
6. 尚未输出 Casimir 能量或力矩结论。
7. 当前 normal / $s_{\pm}$ / $d$-wave local baseline 不应人为产生非零各向异性。
8. gap sign 仍是 gauge-dependent preliminary diagnostic。

## 工程状态

当前已有测试覆盖：

1. pairing matrix dtype、BdG particle-hole symmetry 和零配对极限；
2. normal-state velocity / mass operator；
3. BdG $K_{\mathrm{para}}$、$K_{\mathrm{dia}}$、$K_{\mathrm{total}}$；
4. $\Sigma_{\mathrm{SC}} = \frac{K_{\mathrm{total}}}{\omega_{\mathrm{eV}}}$；
5. local sheet response interface；
6. unit / static / nonlocal response boundary interfaces；
7. Casimir 骨架函数的 smoke-level 行为。

最近一次全量测试结果为

$$
81\ \mathrm{passed}.
$$

因此，当前仓库已经适合作为后续 superconducting response normalization、
非局域 response 和 Casimir torque 物理机制探索的稳定基础层。
