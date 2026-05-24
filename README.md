# LNO327 卡西米尔力矩基础代码

本项目提供一套底层 Python 代码框架，用于研究卡西米尔力矩是否可以区分
La3Ni2O7 中的 `s_pm` 与 `d_wave` 超导配对对称性。

当前范围：

- 正式采用的四轨道双层 normal-state Hamiltonian。
- 初始 `s_pm` 配对矩阵与 `d_wave`/B1g 配对矩阵，配对幅度单位为 eV。
- BdG 矩阵组装。
- 电导张量旋转与各向异性辅助函数。
- 速度顶点，以及以 eV 为能量输入的能带表象 Kubo 电导。
- Dai/Jiang 形式的反射矩阵，以及卡西米尔能量/力矩 integrand。
- smoke-test 脚本与 pytest 测试覆盖。

当前代码仍然只是底层物理与工程基础层，还不是正式的数值模拟层。

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
