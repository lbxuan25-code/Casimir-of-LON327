# 研究计划与仓库边界

本项目的中期目标不是直接得到卡西米尔力矩数值，而是先弄清楚
$\mathrm{La_3Ni_2O_7}$ / LNO327 在 minimal $s_{\pm}$ 与 $d$-wave 配对下的电导响应是否有稳健、
可解释的对称性差异。只有当电导层的物理和数值诊断都稳定后，才把它接到
Casimir torque 框架中。

## 当前优先级

1. **Pairing 与 BdG 基础**
   - 固定 `(dz1, dx1, dz2, dx2)` 基。
   - 维护 `spm` 与 `dwave` 两个 minimal pairing ansatz，分别代表 $s_{\pm}$ 与 $d$-wave / $B_{1g}$ 通道。
   - 检查 BdG Hermiticity、particle-hole spectrum symmetry、零配对极限。

2. **Gap Structure**
   - 在 normal-state band basis 上投影 gap。
   - 在近似 Fermi surface 上检查 gap magnitude、preliminary sign、near-node 分布。
   - 使用 band-resolved 与 tolerance-sensitive 诊断判断 node 是否稳健。

3. **Conductivity Symmetry**
   - normal-state Kubo conductivity 继续作为基线。
   - BdG superconducting response 当前维护 paramagnetic kernel 与 diamagnetic kernel 两个基础层。
   - 当前也提供 $K_{\mathrm{total}}(i\xi) = K_{\mathrm{para}}(i\xi) + K_{\mathrm{dia}}$ 诊断脚本。
   - 当前新增 $\Sigma_{\mathrm{SC}}(i\xi) = \frac{K_{\mathrm{total}}(i\xi)}{\omega_{\mathrm{eV}}}$，仅用于与 normal-state
     $\sigma(i\xi)$ 做虚频轴 response-kernel 对比，要求 Matsubara $n \ge 1$。
   - 当前新增 $\Delta_0\rightarrow 0$ BdG-normal 极限 benchmark，用于检查关闭
     pairing 时 `spm` 与 `dwave` 是否回到共同 BdG normal limit，以及
     $\Sigma_{\mathrm{SC}}$、$K_{\mathrm{para}}/K_{\mathrm{dia}}/K_{\mathrm{total}}$
     是否有限稳定。
   - normal Kubo 与 BdG $\Sigma_{\mathrm{SC}}$ 的归一化和公式结构不同；benchmark
     只检查趋势、比例、对称性和稳定性，不要求逐项完全相等。
   - 当前新增 imaginary-axis response 收敛性 benchmark，系统扫描 `nk`、`eta`
     与 Matsubara index，确认 normal / `spm` / `dwave` 的 response 不是数值网格、
     broadening 或频率截断造成的伪影。
   - 若 response 对 `nk` 或 `eta` 未收敛，或 `spm` / `dwave` 差异只在小 `nk`
     或特定 `eta` 下出现，应先视为数值问题，不能进入正式 Casimir 积分。
   - 当前进一步新增高 `Nk` 聚焦收敛复查，重点检查上一轮暴露的 normal
     low-Matsubara `Nk` 敏感性是否在 `Nk=48/64/80` 缓解，并判断 `spm` / `dwave`
     差异在高 `Nk` 下是否平台化。
   - 若 normal response 在高 `Nk` 仍不稳定，必须继续改进积分/采样；若
     `spm` / `dwave` 差异随 `Nk` 增大趋近 0，则当前 minimal pairing 的 local
     response 差异不应解释为稳健物理差异。
   - 当前新增 normal-state low-Matsubara k-space sampling 诊断层，用于判断
     normal response 的 `Nk` 敏感是否来自 Fermi-surface-sensitive integration。
     `shifted` 与 `average` sampling 只作为数值诊断方案，不改变 Kubo 公式，也不
     替代 uniform baseline。
   - 若 average sampling 明显改善 normal response 收敛，可作为后续
     normal-response benchmark 的推荐采样方式，但必须保留 uniform 对照。
   - 当前进一步新增 normal-state Fermi-surface-sensitive sampling benchmark。
     `multishift_average` 系统扫描 `s x s` shifted meshes 并报告 shift-to-shift
     std；`fs_window_refined` 只在 coarse mesh 的 Fermi-window cells 周围局部加密并
     保持面积权重。两者都只改变数值采样，不改变 normal Kubo 公式，也不替代
     uniform 默认。
   - 若 `multishift_average` 或 `fs_window_refined` 显著改善收敛，可作为后续
     normal-response 推荐采样基准，但必须保留 uniform 对照；若仍不收敛，下一步
     应考虑 contour / tetrahedron Fermi-surface integration。
   - 当前新增 FS-adaptive BZ integration prototype。该方法用 coarse cell 顶点和中心
     能量定位费米面穿过的 BZ cells，或通过
     `fs_window_factor * max(eta,kBT,omega_eV)` 标记近费米窗口 cells，然后只对这些
     cells 做局部细分。所有点按面积权重归一后仍调用原
     `kubo_conductivity_imag_axis`，因此它只改变 quadrature，不改变 Kubo integrand。
   - 若 `fs_adaptive` 随 `refine_factor` 和 `Nk` 收敛，可作为后续 normal-response
     推荐数值基准，同时保留 uniform / multishift 对照；若仍不收敛，则转向
     triangle / contour Fermi-surface integration。
   - 后续进入 Casimir 前仍需系统确认量纲、规范约定与物理解释，并在命名上明确区分 kernel 与 full conductivity。
   - 主要关心 `xx≈yy`、`xy≈0`、C4 对称性破缺、频率依赖与 pairing-kind 差异。

4. **Future Casimir Torque**
   - Casimir 模块目前只作为公式骨架和 smoke check。
   - 当前新增的是 Casimir 前置接口：把 normal-state $\sigma(i\xi)$ 与 BdG
     $\Sigma_{\mathrm{SC}}(i\xi)$ 统一为 local $q=0$ sheet response matrix。
   - 当前已显式补齐三个 Casimir 前置边界接口：中性 sheet-conductivity convention audit、
     $n=0$ Matsubara static policy、nonlocal $q_{\parallel}$ response interface。
   - 当前也提供 local-response 接口链路冒烟测试，用于验证
     $\mathrm{LocalSheetResponse}\rightarrow\sigma_{\alpha\beta}\rightarrow r
     \rightarrow\mathcal{E}_{\mathrm{integrand}}\rightarrow\tau_{\mathrm{integrand}}$
     的工程链路。
   - 该接口仍不做 Matsubara 求和，不输出 Casimir 能量或力矩。
   - Lifshitz 求和形式上包含 $n=0$ 半权重项，但当前 local isotropic baseline
     默认 `n=0 policy = skip`。这是为避免未定义的 superconducting
     zero-frequency conductivity 产生假反射贡献，不是声称 $n=0$ 物理不存在。
   - 当前 $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega$ 只定义于
     Matsubara $n\ge 1$；不允许定义
     $\Sigma_{\mathrm{SC}}(0)=K_{\mathrm{total}}(0)/0$。
   - `extrapolate_from_lowest_matsubara` 只作为数值敏感性估计；
     `use_static_kernel` 只作为 stiffness-like 静态核诊断，不把
     $K_{\mathrm{total}}(0)$ 当作 sheet conductivity 输入 reflection matrix。
   - 当前新增 integrand-level sensitivity 判据：比较 extrapolated $n=0$ proxy
     与 $n\ge 1$ partial Matsubara-sum torque integrand。`skip` 只有在 proxy
     影响低于阈值或属于 negligible zero-baseline 时才可接受。
   - 若 $n=0$ proxy 对 torque 的影响超过阈值，或 reference torque 近零但 proxy
     非零，则不能输出正式 Casimir torque，必须转向 zero-frequency reflection
     model 的物理建模。
   - 在 superconducting conductivity 尚未完成前，不从 Casimir 输出物理结论。
   - 当前单位路径为 $\sigma_{\mathrm{model}}\rightarrow\sigma_{\mathrm{sheet}}^{\mathrm{SI}}
     =(e^2/\hbar)\sigma_{\mathrm{model}}\rightarrow
     \sigma_{\mathrm{reflection}}=\sigma_{\mathrm{sheet}}^{\mathrm{SI}}/\sigma_0$。
   - 正式 Casimir 阶段仍需选择具体物理方案：真实 finite-$q$ response、
     $n=0$ 物理处理，或外场/应变/表面取向等各向异性来源。
   - 未来若引入真实各向异性机制，必须重新推导或明确选择 zero-frequency
     reflection policy，再考虑正式 Matsubara 求和。

## 模块边界

- `model.py`: normal-state Hamiltonian、normal-state velocity。
- `pairing.py`: minimal `spm` / `dwave` pairing 与 BdG Hamiltonian 组装，即 $s_{\pm}$ 与 $d$-wave / $B_{1g}$ 通道。
- `gap_analysis.py`: Fermi-surface gap 投影与 node/sign 诊断。
- `conductivity.py`: normal-state Kubo conductivity 基线。
- `bdg_response.py`: BdG current vertex、imaginary-axis kernels 与 `Sigma_SC` 诊断。
- `response_interface.py`: Casimir 前置 local $q=0$ sheet response 接口。
- `response_units.py`: neutral sheet-conductivity convention 与 reflection-dimensionless conversion 接口。
- `static_response.py`: $n=0$ Matsubara response policy 接口。
- `nonlocal_response.py`: finite-$q_{\parallel}$ response 的接口占位与局域回退。
- `casimir.py`: 未来使用的 reflection / 能量 / 力矩 integrand 骨架。

Normal-state 运行脚本集中在 `scripts/normal_state/`。输出按阶段归档：
`outputs/normal_state/conductivity_imag/`、`outputs/normal_state/conductivity_real/`、
`outputs/normal_state/sampling_convergence/`、
`outputs/normal_state/fs_sensitive_sampling/`、
`outputs/normal_state/fs_adaptive_integration/`、
`outputs/pairing/gap_structure/`、`outputs/bdg/paramagnetic_kernel_imag/`、
`outputs/bdg/diamagnetic_kernel/`、`outputs/bdg/total_kernel_imag/`、
`outputs/bdg/superconducting_response_imag/`、`outputs/response/local_sheet_imag/`、
`outputs/response/unit_audit/`、`outputs/response/static_response/`、
`outputs/response/bdg_normal_limit/`、
`outputs/response/convergence_imag/`、
`outputs/response/high_nk_convergence/`、
`outputs/response/static_policy_comparison/`、
`outputs/response/n0_sensitivity/`、
`outputs/response/nonlocal_interface/`、
`outputs/casimir/` 和 `outputs/smoke/`。
旧的顶层 normal-state 脚本路径只作为兼容 wrapper。

论文草稿输出的组织原则见 `docs/notes/publication_output_guide.md` 与
`outputs/README.md`。当前推荐把 `pairing/`、`normal_state/`、`bdg/` 和
`response/local_sheet_imag/` 作为主要论文素材来源；`smoke/`、当前 `casimir/`、
`response/static_response/` 和 `response/nonlocal_interface/` 只作为方法边界或工程诊断。
新版绘图脚本默认保存 300 dpi PNG，不额外生成 PDF。

## 常用脚本顺序

```bash
python scripts/normal_state/inspect_normal_state_blocks.py --kx 0.0 --ky 0.0
python scripts/inspect_pairing_structure.py --kx 0.2 --ky -0.5 --delta0-eV 0.04
python scripts/inspect_gap_structure.py --kind dwave --delta0 0.04 --nk 80 --energy-window 0.05 --node-tolerance 0.001
python scripts/compute_bdg_paramagnetic_kernel_imag.py --kind spm --delta0 0.04 --nk 24 --temperature 30 --matsubara-index 1
python scripts/diagnose_bdg_diamagnetic_kernel.py --kinds spm dwave --delta0 0.04 --nk 24 --temperature 30
python scripts/diagnose_bdg_total_kernel_imag.py --kinds spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8
python scripts/diagnose_superconducting_response_imag.py --kinds spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8
python scripts/benchmark_bdg_normal_limit.py --kinds spm dwave --delta0-list 0 1e-5 1e-4 1e-3 1e-2 0.04 --nk 16 --temperature 30 --matsubara-index 1
python scripts/convergence_response_imag.py --kinds normal spm dwave --nk-list 8 12 16 24 32 --eta-list 1e-3 5e-4 1e-4 --matsubara-list 1 2 5 10 --temperature 30 --delta0 0.04
python scripts/refine_high_nk_convergence.py --kinds normal spm dwave --nk-list 32 48 64 80 --eta-list 5e-4 1e-4 --matsubara-list 1 2 --temperature 30 --delta0 0.04
python scripts/diagnose_normal_sampling_convergence.py --nk-list 32 48 64 80 96 128 --eta-list 1e-3 5e-4 2e-4 1e-4 --matsubara-list 1 2 5 --temperature 30 --sampling uniform shifted average
python scripts/benchmark_normal_fs_sensitive_sampling.py --nk-list 32 48 64 80 --eta-list 5e-4 2e-4 1e-4 --matsubara-list 1 2 --temperature 30 --shift-grid-list 1 2 4 8 --sampling uniform multishift_average fs_window_refined
python scripts/benchmark_normal_fs_adaptive_integration.py --nk-list 32 48 64 --eta-list 5e-4 2e-4 1e-4 --matsubara-list 1 2 --temperature 30 --refine-factor-list 2 4 6 --fs-window-factor 1.0 --sampling uniform multishift_average fs_adaptive --shift-grid 4
python scripts/compare_local_sheet_response_imag.py --kinds normal spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8
python scripts/audit_response_units.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1
python scripts/diagnose_static_response.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30
python scripts/compare_static_response_policies.py --kinds normal spm dwave --policies skip extrapolate_from_lowest_matsubara use_static_kernel --nk 16 --temperature 30 --delta0 0.04 --eta 0.0001
python scripts/assess_n0_torque_sensitivity.py --kinds normal spm dwave --nk 16 --temperature 30 --delta0 0.04 --eta 0.0001 --reference-matsubara-min 1 --reference-matsubara-max 8 --sensitivity-threshold 0.01 --include-toy-anisotropic-control
python scripts/diagnose_nonlocal_response_interface.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1
python scripts/smoke_casimir_local_response.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1
python scripts/normal_state/compute_normal_state_conductivity_imag.py --nk 48 --matsubara-index 1
```

`scripts/outline_casimir_process.py` 只用于未来 Casimir 流程的接口 smoke check，
不属于当前 conductivity-symmetry 主线。

## 命名原则

- 只有包含 paramagnetic 与 diamagnetic 两部分、并经过相应物理检查的量，才命名为
  `superconducting conductivity`。
- 当前 BdG 响应输出使用 `kernel` 或 `paramagnetic_response`。
- $K_{\mathrm{dia}}$ 是 diamagnetic kernel 诊断；单独输出时不称为完整超导电导。
- $K_{\mathrm{total}}(i\xi)$ 是 BdG total electromagnetic kernel 诊断；仍不直接作为 Casimir 输入。
- $\Sigma_{\mathrm{SC}}(i\xi)$ 是 $\frac{K_{\mathrm{total}}(i\xi)}{\omega_{\mathrm{eV}}}$ 的虚频轴 superconducting
  sheet response kernel，用于和 normal-state $\sigma(i\xi)$ 比较；它不是 real-axis
  optical conductivity，也仍不直接作为 Casimir 输入。
- `LocalSheetResponse` 是 Casimir 前置接口对象；当前 `valid_for_casimir_input=False`
  是有意保守标记，表示它还缺少 $n=0$、真实 finite-$q$ 非局域响应和物理各向异性机制。
- `SheetConductivityConvention`、`StaticResponsePolicy` 和 `NonlocalSheetResponse` 已把这些
  边界变成显式接口状态，但还没有把当前 response 升级为正式 Casimir input。
- 当前 `StaticResponsePolicy` 的推荐用途是：`skip` 作为 local isotropic baseline
  默认；`extrapolate_from_lowest_matsubara` 仅作敏感性估计；`use_static_kernel`
  仅作 stiffness-like 静态核诊断。
- `assess_n0_torque_sensitivity.py` 只做 fixed $k_{\parallel},\phi,\theta$ 的
  integrand-level / partial Matsubara-sum sensitivity，不做完整
  $k_{\parallel}/\phi$ 积分，也不输出正式 Casimir torque 结论。
- `smoke_casimir_local_response.py` 只验证接口链路和 toy anisotropy 控制组，不是
  Casimir Matsubara 求和，也不提供正式能量 / 力矩结论。
- gap sign 诊断目前是 gauge-dependent preliminary diagnostic；更可靠的是
  magnitude、near-node 分布和 symmetry pattern。
