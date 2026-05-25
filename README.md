# LNO327 超导配对、电导对称性与未来卡西米尔力矩

本项目提供一套底层 Python 代码框架，用于研究 $\mathrm{La_3Ni_2O_7}$ / LNO327 在
minimal $s_{\pm}$ 与 $d$-wave 超导配对下的电磁响应。当前重心是先建立
normal-state、BdG、gap structure 与 conductivity symmetry 的可检查基础；
未来再把经过验证的电导张量输入卡西米尔力矩框架，用力矩作为区分超导对称性的
候选方法。

当前研究主线：

1. 固定四轨道 normal-state 模型与两个 minimal pairing ansatz。
2. 检查 BdG 谱、gap magnitude/sign/node 结构。
3. 研究 $s_{\pm}$ 与 $d$-wave 的电导行为，尤其是对称性、各向异性与 off-diagonal 响应。
4. 只有在电导层清楚后，才进入 Casimir torque 的系统计算。

当前范围：

- 正式采用的四轨道双层 normal-state Hamiltonian。
- 初始 $s_{\pm}$ 配对矩阵与 $d$-wave / $B_{1g}$ 配对矩阵，配对幅度单位为 eV。
- BdG 矩阵组装、BdG current vertex 与 paramagnetic kernel 基础层。
- gap structure / near-node 诊断工具。
- normal-state Kubo 电导与电导张量对称性辅助函数。
- Dai/Jiang 形式的反射矩阵与卡西米尔 integrand 骨架，暂作未来阶段接口。
- smoke-test 脚本与 pytest 测试覆盖。

当前代码仍然只是底层物理与工程基础层，还不是正式的数值模拟层。特别地，
BdG paramagnetic kernel 不是完整 superconducting conductivity；diamagnetic term
已作为独立诊断加入；当前也提供
$K_{\mathrm{total}} = K_{\mathrm{para}} + K_{\mathrm{dia}}$ 与
$\Sigma_{\mathrm{SC}}(i\xi) = \frac{K_{\mathrm{total}}(i\xi)}{\omega_{\mathrm{eV}}}$ 的虚频轴诊断，但它们仍不是
real-axis optical conductivity，也尚未作为 Casimir 输入。

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

## Casimir 前置接口

当前新增的前置接口只把 normal-state $\sigma(i\xi)$ 与 BdG
$\Sigma_{\mathrm{SC}}(i\xi)$ 统一整理为 local $q=0$ sheet response matrix，供未来
`reflection_matrix` 接口衔接和对称性比较使用。它不是正式 Casimir 输入，也不会
进行 Matsubara 求和、能量积分或力矩计算。

```bash
python scripts/compare_local_sheet_response_imag.py --kinds normal spm dwave --delta0 0.04 --nk 24 --temperature 30 --matsubara-min 1 --matsubara-max 8 --eta 0.0001 --output-prefix outputs/response/local_sheet_imag/data/local_sheet_response_imag
```

正式 Casimir 阶段仍缺少：

- $n=0$ Matsubara 项处理，尤其是 $\Sigma_{\mathrm{SC}} = \frac{K_{\mathrm{total}}}{\omega_{\mathrm{eV}}}$ 的零频限制。
- SI sheet conductivity 归一化与单位审计。
- 非局域 $q_{\parallel}$ 响应。
- 能产生 torque 的角向各向异性机制。

Casimir local-response plumbing smoke test：

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
Casimir energy 或 torque 结论。

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

## Output organization

Generated outputs are grouped by calculation stage and physical object:

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

- `normal_state/conductivity_imag`: normal-state Kubo baseline on the imaginary axis.
- `normal_state/conductivity_real`: normal-state Kubo baseline on the real-frequency axis.
- `pairing/gap_structure`: projected gap magnitude/sign/near-node diagnostics.
- `bdg/paramagnetic_kernel_imag`: BdG $K_{\mathrm{para}}(i\xi)$ diagnostics only, not full superconducting conductivity.
- `bdg/diamagnetic_kernel`: BdG $K_{\mathrm{dia}}$ diagnostics only.
- `bdg/total_kernel_imag`: BdG $K_{\mathrm{total}}(i\xi) = K_{\mathrm{para}}(i\xi) + K_{\mathrm{dia}}$ diagnostics. Not Casimir input yet.
- `bdg/superconducting_response_imag`: BdG $\Sigma_{\mathrm{SC}}(i\xi) = \frac{K_{\mathrm{total}}(i\xi)}{\omega_{\mathrm{eV}}}$ for $n \ge 1$, used for comparison with normal-state $\sigma(i\xi)$. Not Casimir input yet and not real-axis conductivity.
- `response/local_sheet_imag`: unified local $q=0$ sheet response interface for normal / $s_{\pm}$ / $d$-wave comparisons. It is a pre-Casimir interface, not final Casimir input.
- `casimir`: reserved for future Casimir calculations.
- `smoke`: lightweight plots or arrays used only to verify scripts and interfaces.
- `smoke/casimir_local_response`: Casimir plumbing smoke outputs for local response matrices. Not formal Casimir calculations.

Legacy `outputs/data/` and `outputs/figures/` may appear in old runs; new scripts should write to the staged directories above.

长期任务边界与执行顺序见 [research_plan.md](docs/notes/research_plan.md)。
normal-state 相关运行脚本集中在 `scripts/normal_state/`，输出集中在
`outputs/normal_state/`；旧的 `scripts/compute_normal_state_*.py` 等路径保留为兼容 wrapper。
