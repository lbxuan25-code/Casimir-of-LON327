# Outputs Guide

本目录保存可复现实验数据、诊断图和接口 smoke 输出。默认原则是：

- `data/` 保存 `.npz` 或 `.csv`，用于复算、重画和表格提取。
- `figures/` 保存 300 dpi `.png`，优先用于论文草稿、组会和笔记。
- `smoke/` 和 `casimir/` 当前只用于接口验证，不代表正式物理结论。

## 论文草稿优先级

优先考虑下列目录中的图和数据作为论文草稿素材：

1. `pairing/gap_structure/`：投影 gap 幅值、near-node 与 preliminary sign 诊断。
2. `normal_state/conductivity_imag/` 和 `normal_state/conductivity_real/`：normal-state Kubo 基线。
3. `normal_state/sampling_convergence/`：normal low-Matsubara k-space sampling
   convergence 诊断，用于比较 uniform / shifted / average mesh。
4. `normal_state/fs_sensitive_sampling/`：normal Fermi-surface-sensitive sampling
   benchmark，用于比较 uniform / multishift_average / fs_window_refined。
5. `normal_state/fs_adaptive_integration/`：normal FS-adaptive BZ integration
   prototype，用于比较 uniform / multishift_average / fs_adaptive。
6. `bdg/paramagnetic_kernel_imag/`、`bdg/diamagnetic_kernel/`、`bdg/total_kernel_imag/`：
   BdG kernel 层次诊断。
7. `bdg/superconducting_response_imag/`：仅 $n\ge 1$ 的
   $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega_{\mathrm{eV}}$ 诊断。
8. `response/bdg_normal_limit/`：$\Delta_0\rightarrow 0$ BdG-normal 极限 benchmark，
   用于检查 response 层连续性、有限性、对称性和 kernel 分项稳定性。
9. `response/convergence_imag/`：imaginary-axis response 的 `nk` / `eta` /
   Matsubara-index 收敛性 benchmark，用于识别数值伪影。
10. `response/high_nk_convergence/`：高 `Nk` 聚焦复查，用于判断 normal low-Matsubara
   response 是否在 `Nk=48/64/80` 缓解，以及 `spm` / `dwave` 差异是否平台化。
11. `response/local_sheet_imag/` 与 `response/static_policy_comparison/`：
   Casimir 前置 response 接口和 $n=0$ policy 边界说明。
12. `casimir/local_response_integral/`：local-response Casimir integral benchmark，
   包含 $n\ge 1$ Matsubara 求和、$k_{\parallel}/\phi$ 积分和 $\theta$ 扫描。
13. `casimir/local_response_integral/convergence/`：上述 local-response benchmark 的
   Matsubara、$k_{\parallel}$ cutoff/grid 和 $\phi$ grid 收敛性诊断。
14. `casimir/local_response_integral/final_convergence/`：一键式 local-response integral
   final convergence runner 输出，仍为 benchmark-only，不是正式 Casimir 结论。
15. `casimir/local_response_integral/refined_convergence/`：针对 Matsubara tail 与
   fixed-du clean cutoff 的 refined convergence blocker 诊断，仍不是正式 Casimir 结论。
16. `casimir/local_response_integral/cache/`：local-response sheet tensor cache，
   只用于加速 benchmark，不改变物理公式或积分公式。

## 不应作为论文结论的输出

- `smoke/`：只验证脚本和接口链路。
- `casimir/`：当前预留给未来正式计算。
- `response/static_response/` 和 `response/nonlocal_interface/`：接口边界诊断，不是最终物理方案。
- `response/static_policy_comparison/` 中的 `extrapolate_from_lowest_matsubara` 和
  `use_static_kernel`：只作敏感性或 stiffness-like 静态核诊断。
- `response/n0_sensitivity/`：只作 fixed $k_{\parallel},\phi,\theta$ 下的
  integrand-level partial Matsubara-sum sensitivity；用于判断 `skip` 是否可接受，
  不是完整 Casimir torque 结论。
- `response/convergence_imag/`：只作 response 层数值收敛性诊断；若未收敛，
  不能进入正式 Casimir 积分。
- `response/high_nk_convergence/`：只作高 `Nk` response 收敛复查；若 normal
  response 仍不稳定，不能进入 local-response Casimir 积分。
- `normal_state/sampling_convergence/`：只作 normal-state low-Matsubara sampling
  诊断；shifted / average mesh 不改变 Kubo 公式，也不替代 uniform 对照。
- `normal_state/fs_sensitive_sampling/`：只作 normal-state FS-sensitive sampling
  benchmark；multishift_average / fs_window_refined 只改变数值采样，不改变 Kubo
  公式，也不替代 uniform 默认。若仍不收敛，应考虑 contour / tetrahedron
  Fermi-surface integration，且继续暂停正式 local-response Casimir 积分。
- `normal_state/fs_adaptive_integration/`：只作 normal-state FS-adaptive BZ integration
  prototype；fs_adaptive 只细分 coarse FS cells 并保持面积权重，不改变 Kubo
  integrand，也不是 strict contour / triangle 解析积分。若仍不收敛，下一步转向
  triangle / contour Fermi-surface integration，且继续暂停正式 local-response
  Casimir 积分。
- `casimir/local_response_integral/`：只作 local-response integral benchmark；
  `n0_policy=skip`，`finite_q_resolved=False`，`benchmark_only=True`，不得作为正式
  Casimir energy / torque 结论。
- `casimir/local_response_integral/convergence/`：只作 local-response integral
  convergence benchmark；用于判断数值设置，不改变 `n0_policy=skip` 和
  `finite_q_resolved=False` 边界。

## 当前 n=0 约定

Lifshitz 求和形式上包含 $n=0$ 半权重项。当前 local isotropic baseline 默认
`n=0 policy = skip`，不是因为 $n=0$ 不存在，而是因为 superconducting
$\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega$ 只定义于 $n\ge 1$。
`skip` 只有在 extrapolated $n=0$ proxy sensitivity 低于阈值或为
negligible zero-baseline 时才可接受；若影响超过阈值，必须先推导
zero-frequency reflection model。

不要把

$$
K_{\mathrm{total}}(0)/0
$$

定义为 $\Sigma_{\mathrm{SC}}(0)$，也不要把 $K_{\mathrm{total}}(0)$ 直接作为
sheet conductivity 输入 reflection matrix。
