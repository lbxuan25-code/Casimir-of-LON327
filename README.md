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

## 当前阶段状态

数值稳定性阶段已归纳，详见 `docs/notes/numerical_stability_summary.md`。当前可以进入
local-response distance scan benchmark；但这仍然不是正式 Casimir 结论，仍需保留
`local_response=True`、`finite_q_resolved=False`、`n0_policy=skip`、
`benchmark_only=True` 的边界。

## 当前仓库阅读入口

当前不建议从 `outputs/` 子目录逐个寻找结论，应先阅读阶段报告：

- `docs/reports/current_project_status.md`
- `docs/reports/finite_q_response_status.md`
- `docs/reports/local_response_baseline_status.md`
- `docs/notes/numerical_stability_summary.md`

这些入口区分了 local-response baseline、finite-q response diagnostic prototype 和仍禁止输出
正式 Casimir torque 结论的边界。

当前 active 输出：

- 最新 finite-q 结果：`validation/outputs/response/finite_q_raw_q0_consistency/`
- local-response distance scan：`validation/outputs/casimir/local_response_integral/distance_scan/`

历史诊断结果已归档到 `validation/outputs/archive/`，移动清单见
`validation/outputs/archive/ARCHIVE_INDEX.md`。

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

新分析脚本和 notebook 建议优先从稳定窄入口导入：

```python
from lno327.api import KuboConfig, PairingAmplitudes, local_response_imag_axis
```

包根目录 `lno327` 仍保留历史 re-export，用于兼容已有诊断脚本。

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
python validation/scripts/numerical_stability/benchmark_bdg_normal_limit.py --kinds spm dwave --delta0-list 0 1e-5 1e-4 1e-3 1e-2 0.04 --nk 16 --temperature 30 --matsubara-index 1 --eta 0.0001 --output-prefix validation/outputs/archive/response/bdg_normal_limit/data/bdg_normal_limit
```

该 benchmark 只验证 BdG response 层在关闭 pairing 时是否连续、有限、保持
C4 对称性，并检查 `spm` / `dwave` 是否回到共同 BdG normal limit。normal Kubo
与 BdG $\Sigma_{\mathrm{SC}}$ 的归一化和公式结构不同，因此不要求二者逐项相等。
若出现发散、强不连续或对称性破坏，应先修复 response 层，不能进入 Casimir 积分。

检查 imaginary-axis response 的 `nk` / `eta` / Matsubara-index 收敛性：

```bash
python validation/scripts/numerical_stability/convergence_response_imag.py --kinds normal spm dwave --nk-list 8 12 16 24 32 --eta-list 1e-3 5e-4 1e-4 --matsubara-list 1 2 5 10 --temperature 30 --delta0 0.04 --output-prefix validation/outputs/archive/response/convergence_imag/data/convergence_imag
```

该 benchmark 只检查 response 层数值稳定性，不做 Casimir 结果。若响应对
`nk` 或 `eta` 未收敛，不能进入正式 Casimir 积分；若 `spm` / `dwave` 差异只在
小 `nk` 或特定 `eta` 下出现，应视为数值伪影。

针对高 `Nk` 的聚焦收敛复查：

```bash
python validation/scripts/numerical_stability/refine_high_nk_convergence.py --kinds normal spm dwave --nk-list 32 48 64 80 --eta-list 5e-4 1e-4 --matsubara-list 1 2 --temperature 30 --delta0 0.04 --output-prefix validation/outputs/archive/response/high_nk_convergence/data/high_nk_convergence
```

该复查用于确认上一轮发现的 normal low-Matsubara `Nk` 敏感性是否能在
`Nk=48/64/80` 缓解。若 normal response 在高 `Nk` 仍不稳定，当前不建议进入
local-response Casimir 积分；若 `spm` / `dwave` 差异在高 `Nk` 下趋近 0，则不应
解释为稳健物理差异。

诊断 normal-state low-Matsubara 的 k-space 采样问题：

```bash
python validation/scripts/numerical_stability/diagnose_normal_sampling_convergence.py --nk-list 32 48 64 80 96 128 --eta-list 1e-3 5e-4 2e-4 1e-4 --matsubara-list 1 2 5 --temperature 30 --sampling uniform shifted average --output-prefix validation/outputs/archive/normal_state/sampling_convergence/data/normal_sampling_convergence
```

`shifted` / `average` sampling 只是数值诊断方案，不改变 normal Kubo 公式，也不替代
默认 uniform 结果。若 average sampling 明显改善收敛，可作为后续 normal-response
benchmark 的推荐采样方式，但必须保留 uniform 对照。

建立更系统的 normal-state Fermi-surface-sensitive sampling benchmark：

```bash
python validation/scripts/numerical_stability/benchmark_normal_fs_sensitive_sampling.py --nk-list 32 48 64 80 --eta-list 5e-4 2e-4 1e-4 --matsubara-list 1 2 --temperature 30 --shift-grid-list 1 2 4 8 --sampling uniform multishift_average fs_window_refined --output-prefix validation/outputs/archive/normal_state/fs_sensitive_sampling/data/fs_sensitive_sampling
```

`multishift_average` 对 `s x s` 个 fractional shifted meshes 做平均并报告 shift-to-shift
std；`fs_window_refined` 先用 coarse mesh 找到
`|E_band(k)| < max(eta, kBT, omega_eV)` 的 Fermi-window cells，再在这些局部 cell 内
加密并保持面积权重。两者都只改变数值采样，不改变 Kubo 物理公式，也不替代
uniform 默认。若这些方案仍不收敛，下一步应考虑 contour / tetrahedron
Fermi-surface integration；在 normal response 收敛前仍暂停正式 local-response
Casimir 积分。

运行 FS-adaptive BZ integration prototype：

```bash
python validation/scripts/numerical_stability/benchmark_normal_fs_adaptive_integration.py --nk-list 32 48 64 --eta-list 5e-4 2e-4 1e-4 --matsubara-list 1 2 --temperature 30 --refine-factor-list 2 4 6 --fs-window-factor 1.0 --sampling uniform multishift_average fs_adaptive --shift-grid 4 --output-prefix validation/outputs/archive/normal_state/fs_adaptive_integration/data/fs_adaptive
```

`fs_adaptive` 先用 coarse cells 的顶点和中心能量判断费米面是否穿过该 cell，或是否落入
`fs_window_factor * max(eta, kBT, omega_eV)`，再只对这些 FS cells 做局部
`refine_factor x refine_factor` 细分。所有点按面积权重归一后送入现有
`kubo_conductivity_imag_axis`，因此它不改 Kubo integrand，只改 quadrature。若
`fs_adaptive` 仍不随 `refine_factor` 和 `Nk` 稳定，下一步应转向 triangle / contour
Fermi-surface integration；在 normal response 收敛前仍暂停正式 local-response
Casimir 积分。

## Casimir 前置接口

当前新增的前置接口只把 normal-state $\sigma(i\xi)$ 与 BdG
$\Sigma_{\mathrm{SC}}(i\xi)$ 统一整理为 local $q=0$ sheet response matrix，供未来
`reflection_matrix` 接口衔接和对称性比较使用。它不是正式 Casimir 输入，也不会
进行 Matsubara 求和、能量积分或力矩计算。

```bash
python validation/scripts/response/compare_local_sheet_response_imag.py --kinds normal spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8 --eta 0.0001 --output-prefix validation/outputs/archive/response/local_sheet_imag/data/local_sheet_response_imag
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
python validation/scripts/numerical_stability/audit_response_units.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1
python validation/scripts/numerical_stability/diagnose_static_response.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30
python validation/scripts/response/compare_static_response_policies.py --kinds normal spm dwave --policies skip extrapolate_from_lowest_matsubara use_static_kernel --nk 16 --temperature 30 --delta0 0.04 --eta 0.0001 --distance 3e-8 --k-parallel 1e6 --phi 0.2 --theta 0.7 --output-prefix validation/outputs/archive/response/static_policy_comparison/data/static_policy_comparison
python validation/scripts/numerical_stability/assess_n0_torque_sensitivity.py --kinds normal spm dwave --nk 16 --temperature 30 --delta0 0.04 --eta 0.0001 --reference-matsubara-min 1 --reference-matsubara-max 8 --sensitivity-threshold 0.01 --theta-scan-num 41 --include-toy-anisotropic-control --output-prefix validation/outputs/archive/response/n0_sensitivity/data/n0_sensitivity
python validation/scripts/numerical_stability/diagnose_nonlocal_response_interface.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1 --q-parallel 1e6 --phi 0.2
```

Casimir local-response 接口链路冒烟测试：

```bash
python validation/scripts/smoke/smoke_casimir_local_response.py --kinds normal spm dwave --delta0 0.04 --nk 16 --temperature 30 --matsubara-index 1 --distance 3e-8 --k-parallel 1e6 --phi 0.2 --theta 0.7 --output-prefix validation/outputs/archive/smoke/smoke/casimir_local_response/data/casimir_local_response_smoke
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

Local-response Casimir integral benchmark 入口：

```bash
python validation/scripts/casimir/benchmark_casimir_local_response_integral.py --kinds normal spm dwave --distance-list 3e-8 5e-8 1e-7 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --matsubara-min 1 --matsubara-max 8 --kparallel-num 64 --kparallel-max-factor 20 --phi-num 32 --temperature 30 --normal-nk 96 --normal-eta 1e-4 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --output-prefix validation/outputs/archive/casimir/local_response_integral/data/local_response_integral
python validation/scripts/casimir/converge_casimir_local_response_integral.py --kinds normal spm dwave --distance 5e-8 --matsubara-max-list 2 4 8 16 --kparallel-num-list 16 32 64 --kparallel-max-factor-list 10 20 40 --phi-num-list 16 32 64 --temperature 30 --normal-nk 96 --normal-eta 1e-4 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --output-prefix validation/outputs/archive/casimir/local_response_integral/convergence/data/local_integral_convergence
python validation/scripts/casimir/run_casimir_local_convergence_final.py --dry-run
python validation/scripts/casimir/refine_casimir_local_convergence_blockers.py --dry-run
python validation/scripts/casimir/benchmark_casimir_local_response_distance_scan.py --dry-run
python validation/scripts/finite_q_diagnostics/diagnose_finite_q_response_anisotropy.py
python validation/scripts/finite_q_diagnostics/diagnose_finite_q_local_limit_decomposition.py --quick
```

该 benchmark 做 $n\ge 1$ Matsubara 求和、$k_{\parallel}/\phi$ 积分和 $\theta$ 扫描；
仍使用 local response、跳过 $n=0$、不含 finite-$q$ response，也不输出正式 Casimir 结论。

finite-q response diagnostic 只检查 response 层角向各向异性，`q_magnitude` 使用
dimensionless BZ momentum；当前仍是 prototype，不是最终 gauge-invariant
finite-q Casimir input。
finite-q local-limit decomposition diagnostic 只拆解 finite-q bubble 的 q->0 local
component 对应关系，不接入 Casimir，也不输出 torque 结论。

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

生成输出按计算阶段和物理对象归档。当前 `outputs/` 只承担数据产物职责；阶段报告在
`docs/reports/`，历史结果在 `validation/outputs/archive/`，可复用中间张量在 `validation/cache/`：

输出说明总入口见 [outputs/README.md](outputs/README.md)，论文草稿整理建议见
[publication_output_guide.md](docs/notes/publication_output_guide.md)。新版绘图脚本默认生成
300 dpi PNG，并尽量统一字号、网格和留白；对应 `.npz` / `.csv` 数据应与图片一起保留，
便于后续按论文版式重画或提取表格。

```text
outputs/
  README.md
  normal_state/
    conductivity_imag/
    conductivity_real/
  pairing/
    gap_structure/
  bdg/
    paramagnetic_kernel_imag/
    diamagnetic_kernel/
    total_kernel_imag/
    superconducting_response_imag/
  response/
    finite_q_raw_q0_consistency/
  casimir/
    local_response_integral/
      distance_scan/
  cache/
    casimir_local_response/
      response_tensors/
  archive/
    normal_state/
    response/
    casimir/
    smoke/
```

- `normal_state/conductivity_imag`: normal-state Kubo 虚频轴基线。
- `normal_state/conductivity_real`: normal-state Kubo 实频轴基线。
- `pairing/gap_structure`: 投影 gap 幅值 / 符号 / near-node 诊断。
- `bdg/paramagnetic_kernel_imag`: 仅用于 BdG $K_{\mathrm{para}}(i\xi)$ 诊断，不是完整超导电导。
- `bdg/diamagnetic_kernel`: 仅用于 BdG $K_{\mathrm{dia}}$ 诊断。
- `bdg/total_kernel_imag`: BdG $K_{\mathrm{total}}(i\xi) = K_{\mathrm{para}}(i\xi) + K_{\mathrm{dia}}$ 诊断，目前不是 Casimir 输入。
- `bdg/superconducting_response_imag`: BdG $\Sigma_{\mathrm{SC}}(i\xi) = \frac{K_{\mathrm{total}}(i\xi)}{\omega_{\mathrm{eV}}}$，仅定义于 $n \ge 1$，用于和 normal-state $\sigma(i\xi)$ 比较；目前不是 Casimir 输入，也不是实频轴电导。
- `response/finite_q_raw_q0_consistency`: 当前 finite-q response 主线最新诊断。
- `casimir/local_response_integral/distance_scan`: local-response distance scan baseline。
- `cache/casimir_local_response/response_tensors`: local-response benchmark 复用的 response tensor cache。
- `archive/`: 已完成阶段的历史结果，移动清单见 `validation/outputs/archive/ARCHIVE_INDEX.md`。

旧运行中可能还会出现 legacy `outputs/data/` 和 `outputs/figures/`；新脚本应写入上面的分阶段目录或明确进入 `validation/outputs/archive/`。

长期任务边界与执行顺序见 [research_plan.md](docs/notes/research_plan.md)。
normal-state 相关运行脚本集中在 `scripts/normal_state/`，输出集中在
`outputs/normal_state/`；旧的 `scripts/compute_normal_state_*.py` 等路径保留为兼容 wrapper。
