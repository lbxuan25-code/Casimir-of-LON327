# LNO327 超导配对、电导对称性与未来卡西米尔力矩

本项目提供一套底层 Python 代码框架，用于研究 $\mathrm{La_3Ni_2O_7}$ / LNO327 在
minimal $s_{\pm}$ 与 $d$-wave 超导配对下的电磁响应。当前重心是先建立
normal-state、BdG、gap 结构与电导对称性的可检查基础；
未来再把经过验证的电导张量输入卡西米尔力矩框架，用力矩作为区分超导对称性的
候选方法。

当前研究主线：

1. 固定四轨道 normal-state 模型与两个 minimal pairing ansatz。
2. 检查 BdG 谱、gap 幅值 / 符号 / node 结构。
3. 研究 $s_{\pm}$ 与 $d$-wave 的电导行为，尤其是对称性、各向异性与非对角响应。
4. 只有在电导层清楚后，才进入 Casimir torque 的系统计算。

当前范围：

- 正式采用的四轨道双层 normal-state Hamiltonian。
- 初始 $s_{\pm}$ 配对矩阵与 $d$-wave / $B_{1g}$ 配对矩阵，配对幅度单位为 eV。
- BdG 矩阵组装、BdG current vertex 与 paramagnetic kernel 基础层。
- gap structure / near-node 诊断工具。
- normal-state Kubo 电导与电导张量对称性辅助函数。
- 反射矩阵与卡西米尔 integrand 骨架，暂作未来阶段接口。
- 冒烟测试脚本与 pytest 测试覆盖。

当前代码仍然只是底层物理与工程基础层，还不是正式的数值模拟层。特别地，
BdG paramagnetic kernel 不是完整超导电导；diamagnetic term
已作为独立诊断加入；当前也提供
$K_{\mathrm{total}} = K_{\mathrm{para}} + K_{\mathrm{dia}}$ 与
$\Sigma_{\mathrm{SC}}(i\xi) = \frac{K_{\mathrm{total}}(i\xi)}{\omega_{\mathrm{eV}}}$ 的虚频轴诊断，但它们仍不是
实频轴 optical conductivity，也尚未作为 Casimir 输入。
当前 local isotropic baseline 对 Lifshitz 形式中的 $n=0$ Matsubara 半权重项采用
保守默认：`n=0 policy = skip`。这不是说 $n=0$ 项不存在，而是因为当前
superconducting $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega$ 只定义于
$n\ge 1$，直接构造零频 sheet conductivity 会引入未定义的假贡献。
`extrapolate_from_lowest_matsubara` 只作为数值敏感性估计，
`use_static_kernel` 只作为 stiffness-like 静态核诊断，不作为 sheet conductivity。
进一步地，`skip` 只有在 extrapolated $n=0$ proxy 对 torque integrand 的影响
低于阈值或为 negligible zero-baseline 时才可接受；若 proxy 影响超过阈值，
当前仓库禁止输出正式 Casimir torque 结论，必须先建立 zero-frequency reflection model。

最小序参量采用四轨道基 `(dz1, dx1, dz2, dx2)`，配对幅度记为
`delta0_eV`：

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

$$
\Delta_{d}(\mathbf{k})
= \Delta_0 \left[\cos(k_x) + \cos(k_y)\right]
\begin{pmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{pmatrix}.
$$

运行测试：

```bash
pytest
```

检查单个动量点：

```bash
python scripts/normal_state/inspect_normal_state_blocks.py --kx 0.0 --ky 0.0
```

检查 pairing 结构和 BdG 谱：

```bash
python scripts/inspect_pairing_structure.py --kx 0.2 --ky -0.5 --delta0-eV 0.04
```

检查 normal-state 费米面附近的投影 gap 结构：

```bash
python scripts/inspect_gap_structure.py --kind spm --delta0 0.04 --nk 80 --energy-window 0.05 --node-tolerance 0.001 --output-prefix outputs/pairing/gap_structure/data/gap_structure_spm
```

检查 BdG paramagnetic kernel 的虚频轴基础响应：

```bash
python scripts/compute_bdg_paramagnetic_kernel_imag.py --kind spm --delta0 0.04 --nk 24 --temperature 30 --matsubara-index 1
```

扫描 BdG paramagnetic kernel 对称性诊断：

```bash
python scripts/diagnose_bdg_paramagnetic_kernel.py --kinds spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8 --eta 0.0001 --output-prefix outputs/bdg/paramagnetic_kernel_imag/data/K_para_imag
```

诊断 BdG diamagnetic kernel：

```bash
python scripts/diagnose_bdg_diamagnetic_kernel.py --kinds spm dwave --delta0 0.04 --nk 24 --temperature 30 --output-prefix outputs/bdg/diamagnetic_kernel/data/K_dia
```

扫描 BdG total kernel：

```bash
python scripts/diagnose_bdg_total_kernel_imag.py --kinds spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8 --eta 0.0001 --output-prefix outputs/bdg/total_kernel_imag/data/K_total_imag
```

扫描 BdG superconducting response kernel $\Sigma_{\mathrm{SC}} = \frac{K_{\mathrm{total}}}{\omega_{\mathrm{eV}}}$：

```bash
python scripts/diagnose_superconducting_response_imag.py --kinds spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8 --eta 0.0001 --output-prefix outputs/bdg/superconducting_response_imag/data/Sigma_SC_imag
```

检查 $\Delta_0\rightarrow 0$ 的 BdG-normal 极限：

```bash
python scripts/benchmark_bdg_normal_limit.py --kinds spm dwave --delta0-list 0 1e-5 1e-4 1e-3 1e-2 0.04 --nk 16 --temperature 30 --matsubara-index 1 --eta 0.0001 --output-prefix outputs/response/bdg_normal_limit/data/bdg_normal_limit
```

该 benchmark 只验证 BdG response 层在关闭 pairing 时是否连续、有限、保持
C4 对称性，并检查 `spm` / `dwave` 是否回到共同 BdG normal limit。normal Kubo
与 BdG $\Sigma_{\mathrm{SC}}$ 的归一化和公式结构不同，因此不要求二者逐项相等。
若出现发散、强不连续或对称性破坏，应先修复 response 层，不能进入 Casimir 积分。

检查 imaginary-axis response 的 `nk` / `eta` / Matsubara-index 收敛性：

```bash
python scripts/convergence_response_imag.py --kinds normal spm dwave --nk-list 8 12 16 24 32 --eta-list 1e-3 5e-4 1e-4 --matsubara-list 1 2 5 10 --temperature 30 --delta0 0.04 --output-prefix outputs/response/convergence_imag/data/convergence_imag
```

该 benchmark 只检查 response 层数值稳定性，不做 Casimir 结果。若响应对
`nk` 或 `eta` 未收敛，不能进入正式 Casimir 积分；若 `spm` / `dwave` 差异只在
小 `nk` 或特定 `eta` 下出现，应视为数值伪影。

针对高 `Nk` 的聚焦收敛复查：

```bash
python scripts/refine_high_nk_convergence.py --kinds normal spm dwave --nk-list 32 48 64 80 --eta-list 5e-4 1e-4 --matsubara-list 1 2 --temperature 30 --delta0 0.04 --output-prefix outputs/response/high_nk_convergence/data/high_nk_convergence
```

该复查用于确认上一轮发现的 normal low-Matsubara `Nk` 敏感性是否能在
`Nk=48/64/80` 缓解。若 normal response 在高 `Nk` 仍不稳定，当前不建议进入
local-response Casimir 积分；若 `spm` / `dwave` 差异在高 `Nk` 下趋近 0，则不应
解释为稳健物理差异。

## Casimir 前置接口

当前新增的前置接口只把 normal-state $\sigma(i\xi)$ 与 BdG
$\Sigma_{\mathrm{SC}}(i\xi)$ 统一整理为 local $q=0$ sheet response matrix，供未来
`reflection_matrix` 接口衔接和对称性比较使用。它不是正式 Casimir 输入，也不会
进行 Matsubara 求和、能量积分或力矩计算。

```bash
python scripts/compare_local_sheet_response_imag.py --kinds normal spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8 --eta 0.0001 --output-prefix outputs/response/local_sheet_imag/data/local_sheet_response_imag
```

当前采用中性命名的 sheet-conductivity convention。反射矩阵前的单位路径为

$$
\sigma_{\mathrm{model}}
\rightarrow
\sigma_{\mathrm{sheet}}^{\mathrm{SI}}
= \frac{e^2}{\hbar}\sigma_{\mathrm{model}}
\rightarrow
\sigma_{\mathrm{reflection}}
= \frac{\sigma_{\mathrm{sheet}}^{\mathrm{SI}}}{\sigma_0},
\qquad
\sigma_0=\sqrt{\epsilon_0/\mu_0}.
$$

正式 Casimir 阶段仍需要选择物理方案：

- 为 $n=0$ Matsubara 项选择最终处理方案；Lifshitz 求和形式上包含 $n=0$
  半权重，但当前 local isotropic baseline 默认 `skip`，避免把未定义的
  superconducting zero-frequency conductivity 作为 reflection matrix 输入。
- `extrapolate_from_lowest_matsubara` 只用于数值敏感性估计；
  `use_static_kernel` 只输出 stiffness-like $K_{\mathrm{total}}(0)$ 诊断，不定义
  $\Sigma_{\mathrm{SC}}(0)=K_{\mathrm{total}}(0)/0$，也不把
  $K_{\mathrm{total}}(0)$ 直接当作 sheet conductivity。
- 当前已建立 SI sheet conductivity 转换层，但仍需决定它如何和完整 $n=0$、finite-$q$ 以及真实各向异性机制一起作为正式 Casimir 输入。
- 实现真实非局域 $q_{\parallel}$ 响应；当前已有局域回退和 finite-$q$ 显式占位。
- 能产生 torque 的角向各向异性机制。

单位、静态项和 nonlocal 接口诊断：

```bash
python scripts/audit_response_units.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1
python scripts/diagnose_static_response.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30
python scripts/compare_static_response_policies.py --kinds normal spm dwave --policies skip extrapolate_from_lowest_matsubara use_static_kernel --nk 16 --temperature 30 --delta0 0.04 --eta 0.0001 --distance 3e-8 --k-parallel 1e6 --phi 0.2 --theta 0.7 --output-prefix outputs/response/static_policy_comparison/data/static_policy_comparison
python scripts/assess_n0_torque_sensitivity.py --kinds normal spm dwave --nk 16 --temperature 30 --delta0 0.04 --eta 0.0001 --reference-matsubara-min 1 --reference-matsubara-max 8 --sensitivity-threshold 0.01 --theta-scan-num 41 --include-toy-anisotropic-control --output-prefix outputs/response/n0_sensitivity/data/n0_sensitivity
python scripts/diagnose_nonlocal_response_interface.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1 --q-parallel 1e6 --phi 0.2
```

Casimir local-response 接口链路冒烟测试：

```bash
python scripts/smoke_casimir_local_response.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1 --distance 3e-8 --k-parallel 1e6 --phi 0.2 --theta 0.7 --output-prefix outputs/smoke/casimir_local_response/data/casimir_local_response_smoke
```

该脚本只验证

$$
\mathrm{LocalSheetResponse}
\rightarrow \sigma_{\alpha\beta}
\rightarrow r
\rightarrow \mathcal{E}_{\mathrm{integrand}}
\rightarrow \tau_{\mathrm{integrand}}
$$

的工程链路；它包含各向同性与 toy anisotropic response 控制组，但不输出正式
Casimir 能量或力矩结论。

绘制 normal-state 能带：

```bash
python scripts/normal_state/inspect_band_structure.py
```

计算 normal-state 虚频轴电导：

```bash
python scripts/normal_state/compute_normal_state_conductivity_imag.py --nk 48 --matsubara-index 1
```

计算 normal-state 实频轴电导扫描：

```bash
python scripts/normal_state/compute_normal_state_conductivity_real.py --nk 48 --omega-min 0.01 --omega-max 0.5 --num-omega 100 --eta 0.001 --output-prefix outputs/normal_state/conductivity_real/data/normal_state_conductivity_real
```

## 输出组织

生成输出按计算阶段和物理对象归档：

输出说明总入口见 [outputs/README.md](outputs/README.md)，论文草稿整理建议见
[publication_output_guide.md](docs/notes/publication_output_guide.md)。新版绘图脚本默认生成
300 dpi PNG，并尽量统一字号、网格和留白；对应 `.npz` / `.csv` 数据应与图片一起保留，
便于后续按论文版式重画或提取表格。

```text
outputs/
  normal_state/
    conductivity_imag/
      data/
      figures/
    conductivity_real/
      data/
      figures/
  pairing/
    gap_structure/
      data/
      figures/
  bdg/
    paramagnetic_kernel_imag/
      data/
      figures/
    diamagnetic_kernel/
      data/
      figures/
    total_kernel_imag/
      data/
      figures/
    superconducting_response_imag/
      data/
      figures/
  response/
    local_sheet_imag/
      data/
      figures/
    bdg_normal_limit/
      data/
      figures/
    convergence_imag/
      data/
      figures/
    high_nk_convergence/
      data/
      figures/
    unit_audit/
      data/
      figures/
    static_response/
      data/
      figures/
    static_policy_comparison/
      data/
      figures/
    n0_sensitivity/
      data/
      figures/
    nonlocal_interface/
      data/
      figures/
  casimir/
    data/
    figures/
  smoke/
    data/
    figures/
    casimir_local_response/
      data/
      figures/
```

- `normal_state/conductivity_imag`: normal-state Kubo 虚频轴基线。
- `normal_state/conductivity_real`: normal-state Kubo 实频轴基线。
- `pairing/gap_structure`: 投影 gap 幅值 / 符号 / near-node 诊断。
- `bdg/paramagnetic_kernel_imag`: 仅用于 BdG $K_{\mathrm{para}}(i\xi)$ 诊断，不是完整超导电导。
- `bdg/diamagnetic_kernel`: 仅用于 BdG $K_{\mathrm{dia}}$ 诊断。
- `bdg/total_kernel_imag`: BdG $K_{\mathrm{total}}(i\xi) = K_{\mathrm{para}}(i\xi) + K_{\mathrm{dia}}$ 诊断，目前不是 Casimir 输入。
- `bdg/superconducting_response_imag`: BdG $\Sigma_{\mathrm{SC}}(i\xi) = \frac{K_{\mathrm{total}}(i\xi)}{\omega_{\mathrm{eV}}}$，仅定义于 $n \ge 1$，用于和 normal-state $\sigma(i\xi)$ 比较；目前不是 Casimir 输入，也不是实频轴电导。
- `response/local_sheet_imag`: normal / $s_{\pm}$ / $d$-wave 的统一 local $q=0$ sheet response 接口；这是 Casimir 前置接口，不是最终 Casimir 输入。
- `response/bdg_normal_limit`: $\Delta_0\rightarrow 0$ 的 BdG response benchmark；
  检查 pairing 关闭时 `spm` / `dwave` 是否趋同、ratio 是否有限、kernel 分项是否稳定。
  这不是 Casimir 结果，也不要求 BdG response 与 normal Kubo 逐项相等。
- `response/convergence_imag`: imaginary-axis response 数值收敛性 benchmark；
  检查 `nk`、`eta`、Matsubara index 对 normal / `spm` / `dwave` local response 的影响。
  当前仍不包含 finite-$q$ nonlocal response，也不是 Casimir 结果。
- `response/high_nk_convergence`: 高 `Nk` 聚焦复查；重点检查 normal response 与低
  Matsubara index 在 `Nk=32..80` 的稳定性，以及 `spm` / `dwave` 差异是否在高
  `Nk` 下平台化或继续趋近 0。
- `response/unit_audit`: reflection input 前的单位约定和归一化状态诊断。
- `response/static_response`: $n=0$ Matsubara policy 诊断。
- `response/static_policy_comparison`: 当前保守 $n=0$ policy 对比；local baseline
  推荐 `skip`，extrapolate/static-kernel 仅作诊断，不输出正式 Casimir torque 结论。
- `response/n0_sensitivity`: 固定 $k_{\parallel},\phi,\theta$ 下的 integrand-level
  partial Matsubara-sum sensitivity；比较 extrapolated $n=0$ proxy 与 $n\ge 1$
  baseline 的大小，用于判断当前 `skip` 是否可接受。该目录不是完整
  $k_{\parallel}/\phi$ 积分后的 Casimir 结论。
- `response/nonlocal_interface`: nonlocal response 接口诊断；当前未实现真实 finite-$q_{\parallel}$ 物理。
- `casimir`: 预留给未来 Casimir 计算。
- `smoke`: 只用于验证脚本和接口的轻量输出。
- `smoke/casimir_local_response`: local response matrix 接入 Casimir integrand 的 smoke 输出，不是正式 Casimir 计算。

旧运行中可能还会出现 legacy `outputs/data/` 和 `outputs/figures/`；新脚本应写入上面的分阶段目录。

长期任务边界与执行顺序见 [research_plan.md](docs/notes/research_plan.md)。
normal-state 相关运行脚本集中在 `scripts/normal_state/`，输出集中在
`outputs/normal_state/`；旧的 `scripts/compute_normal_state_*.py` 等路径保留为兼容 wrapper。
