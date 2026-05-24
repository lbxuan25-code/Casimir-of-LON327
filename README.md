# LNO327 超导配对、电导对称性与未来卡西米尔力矩

本项目提供一套底层 Python 代码框架，用于研究 La3Ni2O7/LNO327 在
minimal `s_pm` 与 `d_wave` 超导配对下的电磁响应。当前重心是先建立
normal-state、BdG、gap structure 与 conductivity symmetry 的可检查基础；
未来再把经过验证的电导张量输入卡西米尔力矩框架，用力矩作为区分超导对称性的
候选方法。

当前研究主线：

1. 固定四轨道 normal-state 模型与两个 minimal pairing ansatz。
2. 检查 BdG 谱、gap magnitude/sign/node 结构。
3. 研究 `s_pm` 与 `d_wave` 的电导行为，尤其是对称性、各向异性与 off-diagonal 响应。
4. 只有在电导层清楚后，才进入 Casimir torque 的系统计算。

当前范围：

- 正式采用的四轨道双层 normal-state Hamiltonian。
- 初始 `s_pm` 配对矩阵与 `d_wave`/B1g 配对矩阵，配对幅度单位为 eV。
- BdG 矩阵组装、BdG current vertex 与 paramagnetic kernel 基础层。
- gap structure / near-node 诊断工具。
- normal-state Kubo 电导与电导张量对称性辅助函数。
- Dai/Jiang 形式的反射矩阵与卡西米尔 integrand 骨架，暂作未来阶段接口。
- smoke-test 脚本与 pytest 测试覆盖。

当前代码仍然只是底层物理与工程基础层，还不是正式的数值模拟层。特别地，
BdG paramagnetic kernel 不是完整 superconducting conductivity；diamagnetic term
和完整超导电导仍属于后续工作。

最小序参量采用四轨道基 `(dz1, dx1, dz2, dx2)`，配对幅度记为
`delta0_eV`：

```text
Delta_s_pm = delta0_eV * [[0, 0, 1, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 0]]
```

```text
Delta_dwave = delta0_eV * (cos(kx) + cos(ky))
              * [[0, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]]
```

运行测试：

```bash
pytest
```

检查单个动量点：

```bash
python scripts/inspect_normal_state_blocks.py --kx 0.0 --ky 0.0
```

检查 pairing 结构和 BdG 谱：

```bash
python scripts/inspect_pairing_structure.py --kx 0.2 --ky -0.5 --delta0-eV 0.04
```

检查 normal-state 费米面附近的投影 gap 结构：

```bash
python scripts/inspect_gap_structure.py --kind spm --delta0 0.04 --nk 80 --energy-window 0.05 --node-tolerance 0.001
```

检查 BdG paramagnetic kernel 的虚频轴基础响应：

```bash
python scripts/compute_bdg_paramagnetic_kernel_imag.py --kind spm --delta0 0.04 --nk 24 --temperature 30 --matsubara-index 1
```

绘制 normal-state 能带：

```bash
python scripts/inspect_band_structure.py
```

计算 normal-state 虚频轴电导：

```bash
python scripts/compute_normal_state_conductivity_imag.py --nk 48 --matsubara-index 1
```

计算 normal-state 实频轴电导扫描：

```bash
python scripts/compute_normal_state_conductivity_real.py --nk 48 --omega-min 0.01 --omega-max 0.5 --num-omega 100 --eta 0.001 --output-prefix outputs/data/normal_state_conductivity_real
```

长期任务边界与执行顺序见 [research_plan.md](docs/notes/research_plan.md)。
