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

图像与数据输出按论文草稿工作流整理在 `outputs/` 下；总说明见
`outputs/README.md`，图像与叙事建议见 `docs/notes/publication_output_guide.md`。
当前推荐把 gap structure、normal-state conductivity、BdG kernel /
$\Sigma_{\mathrm{SC}}$ 和 local sheet response 图作为主要素材；smoke 与 Casimir
接口输出只用于方法边界说明。

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
为了判断这个 `skip` 是否只是保守还是会隐藏主要 torque 贡献，当前新增了
integrand-level sensitivity 判据：在固定 $k_{\parallel},\phi,\theta$ 下，将
extrapolated $n=0$ proxy torque integrand 与 $n\ge 1$ partial Matsubara-sum
baseline 比较。若 ratio 低于阈值，或两者均为数值零，则当前 local baseline
可继续使用 `skip`；若 ratio 超过阈值，或 reference torque 近零但 proxy 非零，
则禁止输出正式 Casimir torque，必须先推导 zero-frequency reflection model。

当前还新增了 $\Delta_0\rightarrow 0$ BdG-normal 极限 benchmark。该 benchmark
只检查 response 层：当 pairing amplitude 关闭时，minimal `spm` 与 `dwave`
应回到共同的 BdG normal limit，$\Sigma_{\mathrm{SC}}$ 和
$K_{\mathrm{para}}/K_{\mathrm{dia}}/K_{\mathrm{total}}$ 应保持有限稳定，并继续满足
C4 对称性。normal-state Kubo $\sigma(i\xi)$ 与 BdG
$\Sigma_{\mathrm{SC}}(i\xi)$ 的归一化和公式结构不同，因此这里不要求二者逐项相等。
若 benchmark 出现非连续、发散或强烈对称性破坏，应先修复 response 层，不能进入
Casimir 积分。

当前还新增了 imaginary-axis response 收敛性 benchmark，系统扫描 `nk`、`eta`
和 Matsubara index 对 normal / `spm` / `dwave` local response 的影响。该 benchmark
只检查 response 层数值稳定性和 `spm` / `dwave` 差异是否稳健，不做 Casimir 结论。
若 response 随 `nk` 增大未稳定，或随 `eta` 减小出现异常发散 / 随机震荡，则不能进入
正式 Casimir 积分。若 `spm` / `dwave` 差异只在小 `nk` 或特定 `eta` 下出现，应视为
数值伪影。

针对上一轮发现的 normal response 与低 Matsubara index 的 `Nk` 敏感性，当前新增
高 `Nk` 聚焦复查。该复查比较 `Nk=32,48,64,80`，重点看最大 `Nk` 参考变化、
最后两个网格点 `Nk=64\rightarrow 80` 的相对变化，以及每个 `Nk` 的 eta sensitivity。
若 normal response 在高 `Nk` 仍不稳定，不能进入 local-response Casimir 积分；
若 `spm` / `dwave` 差异随高 `Nk` 继续趋近 0，则当前 minimal pairing 的 local
response 差异不应解释为稳健物理差异。只有当高 `Nk` 下仍有稳定平台化差异时，
才适合作为后续物理分析对象。

当前进一步新增 normal-state low-Matsubara k-space sampling 诊断层，专门检查
normal response 的 `Nk` 敏感是否来自 Fermi-surface-sensitive integration。该脚本复用
现有 normal Kubo 公式，并只在脚本内部构造 shifted / averaged mesh。`shifted` 和
`average` sampling 是数值诊断方案，不替代默认 uniform 结果。若 average sampling
显著改善收敛，可作为后续正式 normal-response benchmark 的推荐采样方式，但必须保留
uniform 对照；若 normal response 仍不稳定，则不能进入 local-response Casimir 积分。

在此基础上，当前又新增 normal-state Fermi-surface-sensitive sampling benchmark。
该 benchmark 比较 uniform、`multishift_average` 和 `fs_window_refined`：前者系统扫描
`s x s` shifted meshes 并报告 shift-to-shift std，后者在 coarse mesh 找到
`|E_{\mathrm{band}}(k)| < max(\eta,k_BT,\omega_{\mathrm{eV}})` 的 Fermi-window cells 后局部加密，
并用面积权重近似积分。两种新方案都只改变数值采样，不改变 normal Kubo 物理公式，也不替代
uniform 默认。若这些方案显著改善收敛，可作为后续 normal-response 推荐采样基准；
若仍不收敛，下一步应考虑 contour / tetrahedron Fermi-surface integration。在 normal
response 收敛前，仍暂停正式 local-response Casimir 积分。

当前进一步实现了 FS-adaptive BZ integration prototype。该 prototype 先用 coarse
cell 顶点和中心能量判断费米面是否穿过 cell，或是否进入
`fs_window_factor * max(\eta,k_BT,\omega_{\mathrm{eV}})`，再仅对这些 FS cells 做
`refine_factor x refine_factor` 局部细分。非 FS cells 保留 coarse midpoint；所有点按
面积权重归一，并继续调用现有 `kubo_conductivity_imag_axis`。因此它不改变 normal Kubo
公式，只改变 k-space quadrature。若 `fs_adaptive` 随 `Nk` 和 `refine_factor` 稳定，
可作为后续 normal response 推荐数值基准，同时保留 uniform / multishift 对照；
若仍不稳定，下一步应转向 triangle / contour Fermi-surface integration。在此之前仍不
进入正式 local-response Casimir 积分。

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
新增的 n0 sensitivity 输出也仍是 partial-sum / integrand-level 判断，不包含完整
$k_{\parallel}$ 与 $\phi$ 积分，因此不能替代正式 Lifshitz 计算。

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
   `skip` 的可接受性现在由 n0 proxy sensitivity 诊断决定：低于阈值或
   negligible zero-baseline 时可接受；超过阈值时必须进入 zero-frequency
   reflection model 建模。
3. 有限动量 response 原型已从当前代码主线移除。当前 Casimir 前置接口只覆盖
   local $q=0$ sheet response；后续若重启有限动量方向，需重新设计和验证。

当前阶段仍有以下明确限制：

1. $\Sigma_{\mathrm{SC}}(i\xi_n)$ 只定义在 $n\ge 1$。
2. Lifshitz 求和形式上包含 $n=0$ 半权重，但当前不做正式 Casimir Matsubara 求和。
3. `LocalSheetResponse.valid_for_casimir_input=False` 是有意的保守标记。
4. 当前 response 只包含 local $q=0$ 路径，不包含有限动量计算。
5. 中性 sheet-conductivity convention 已建立，但它还需要与 $n=0$ 物理方案共同组成正式 Casimir 输入。
6. 尚未输出 Casimir 能量或力矩结论。
7. 当前 normal / $s_{\pm}$ / $d$-wave local baseline 不应人为产生非零各向异性。
8. gap sign 仍是 gauge-dependent preliminary diagnostic。

## 工程状态

当前已有测试覆盖：

1. pairing matrix dtype、BdG particle-hole symmetry 和零配对极限；
2. normal-state velocity / mass operator；
3. BdG $K_{\mathrm{para}}$、$K_{\mathrm{dia}}$、$K_{\mathrm{total}}$；
4. $\Sigma_{\mathrm{SC}} = \frac{K_{\mathrm{total}}}{\omega_{\mathrm{eV}}}$；
5. $\Delta_0\rightarrow 0$ BdG-normal 极限 benchmark；
6. imaginary-axis response 的 `nk` / `eta` / Matsubara-index 收敛性 benchmark；
7. 高 `Nk` 聚焦收敛复查；
8. normal-state low-Matsubara sampling convergence 诊断；
9. normal-state FS-sensitive sampling benchmark；
10. normal-state FS-adaptive BZ integration prototype；
11. local sheet response interface；
12. unit / static response boundary interfaces；
13. Casimir 骨架函数的 smoke-level 行为。

最近一次全量测试结果为

$$
118\ \mathrm{passed}.
$$

因此，当前仓库已经适合作为后续 superconducting response normalization、
非局域 response 和 Casimir torque 物理机制探索的稳定基础层。
