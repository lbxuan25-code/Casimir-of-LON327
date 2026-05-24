# 研究计划与仓库边界

本项目的中期目标不是直接得到卡西米尔力矩数值，而是先弄清楚
La3Ni2O7/LNO327 在 minimal `s_pm` 与 `d_wave` 配对下的电导响应是否有稳健、
可解释的对称性差异。只有当电导层的物理和数值诊断都稳定后，才把它接到
Casimir torque 框架中。

## 当前优先级

1. **Pairing 与 BdG 基础**
   - 固定 `(dz1, dx1, dz2, dx2)` 基。
   - 维护 `spm` 与 `dwave` 两个 minimal pairing ansatz。
   - 检查 BdG Hermiticity、particle-hole spectrum symmetry、零配对极限。

2. **Gap Structure**
   - 在 normal-state band basis 上投影 gap。
   - 在近似 Fermi surface 上检查 gap magnitude、preliminary sign、near-node 分布。
   - 使用 band-resolved 与 tolerance-sensitive 诊断判断 node 是否稳健。

3. **Conductivity Symmetry**
   - normal-state Kubo conductivity 继续作为基线。
   - BdG superconducting response 先只维护 paramagnetic kernel 基础层。
   - 后续再系统加入 diamagnetic term，并在命名上明确区分 kernel 与 full conductivity。
   - 主要关心 `xx≈yy`、`xy≈0`、C4 对称性破缺、频率依赖与 pairing-kind 差异。

4. **Future Casimir Torque**
   - Casimir 模块目前只作为公式骨架和 smoke check。
   - 在 superconducting conductivity 尚未完成前，不从 Casimir 输出物理结论。

## 模块边界

- `model.py`: normal-state Hamiltonian、normal-state velocity。
- `pairing.py`: minimal `spm` / `dwave` pairing 与 BdG Hamiltonian 组装。
- `gap_analysis.py`: Fermi-surface gap 投影与 node/sign 诊断。
- `conductivity.py`: normal-state Kubo conductivity 基线。
- `bdg_response.py`: BdG current vertex 与 imaginary-axis paramagnetic kernel。
- `casimir.py`: 未来使用的 reflection / energy / torque integrand 骨架。

## 常用脚本顺序

```bash
python scripts/inspect_normal_state_blocks.py --kx 0.0 --ky 0.0
python scripts/inspect_pairing_structure.py --kx 0.2 --ky -0.5 --delta0-eV 0.04
python scripts/inspect_gap_structure.py --kind dwave --delta0 0.04 --nk 80 --energy-window 0.05 --node-tolerance 0.001
python scripts/compute_bdg_paramagnetic_kernel_imag.py --kind spm --delta0 0.04 --nk 24 --temperature 30 --matsubara-index 1
python scripts/compute_normal_state_conductivity_imag.py --nk 48 --matsubara-index 1
```

`scripts/outline_casimir_process.py` 只用于未来 Casimir 流程的接口 smoke check，
不属于当前 conductivity-symmetry 主线。

## 命名原则

- 只有包含 paramagnetic 与 diamagnetic 两部分、并经过相应物理检查的量，才命名为
  `superconducting conductivity`。
- 当前 BdG 响应输出使用 `kernel` 或 `paramagnetic_response`。
- gap sign 诊断目前是 gauge-dependent preliminary diagnostic；更可靠的是
  magnitude、near-node 分布和 symmetry pattern。
