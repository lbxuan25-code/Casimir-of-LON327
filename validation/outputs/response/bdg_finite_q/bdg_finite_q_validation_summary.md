# finite-q BdG superconducting response 数值检验

## 检验目的

确认当前 finite-q BdG superconducting response 是否具备作为 formal finite-q Casimir input 的条件，并明确哪些子检验只支持 diagnostic use。

## 被检验对象

- generic finite-q engine；
- `PairingAnsatz` input layer；
- finite-q BdG bare current-current kernel；
- amplitude / phase collective-channel Schur restoration；
- `Delta -> 0` normal limit；
- `q -> 0` local limit；
- BdG response 到 reflection input 的候选接口。

## 检验方法与判据

- 裸 kernel 检查 current vertex、Hermiticity、直接 kernel 数值 sanity 和有限性。
- 集体模检查 amplitude / phase collective vertices、Schur complement、mixed block 和 Ward residual。
- normal limit 检查 `Delta0 -> 0` 时 BdG 路径是否回到 normal backend。
- local limit 检查 finite-q 路径在 `q -> 0` 时是否与 local comparison 连续。
- reflection input candidate 检查 BdG response 转为 reflection input 的 gate 是否允许正式使用。
- 本检验涉及 Ward validation，但不涉及 unit conversion policy 或 `n=0` policy 的最终接受。

## 当前有限 q BdG 数值检验状态

### 裸有限 q BdG kernel 检验

状态：通过。

说明：该检验确认有限 q BdG bare kernel 在当前输入参数下可计算、有限，并满足基础数值一致性要求。它本身不证明 gauge-closed finite-q conductivity。

### 振幅/相位集体模 Schur 修正 Ward 检验

状态：未通过。

说明：amplitude / phase collective-channel Schur restoration 尚未使相关 Ward residual 达到可作为 production finite-q conductivity 的标准。部分 commensurate-q 或 LSQ diagnostic 能改善局部 residual，但不能作为 production response repair。

### `Delta -> 0` normal limit 检验

状态：通过。

说明：在 `Delta0 -> 0` 极限下，有限 q BdG 路径与 normal-state backend 的结果一致或在设定容差内一致。

### `q -> 0` local limit 检验

状态：通过。

说明：有限 q 路径在 `q -> 0` 方向上没有暴露新的数值不连续问题；但这不等于完整 gauge-closed finite-q conductivity。

### reflection input 候选接口检验

状态：未通过。

说明：reflection input candidate 可以生成有限张量，但 upstream finite-q BdG validation 尚未通过，因此不能作为 formal Casimir input。

## 主要结果

- finite-q BdG engine 和 pairing input layer 的结构分离可作为实现层证据。
- bare kernel、normal limit、`q -> 0` local limit 是当前通过项。
- collective-sector Ward restoration 和 reflection input gate 是当前阻塞项。
- raw finite-q BdG response 仍是 diagnostic-only。

## 当前判定

未通过：当前结果不能支撑 finite-q BdG response 作为正式 Casimir input。

## 对主流程的影响

- 不阻塞 local `q=0` response。
- 阻塞 raw finite-q BdG response 进入 formal finite-q Casimir input。
- 阻塞基于 BdG finite-q response 的 formal reflection / Casimir pipeline。
- 不改变 normal-state response convention 或单位换算结论。

## 边界说明

- `diagnostic_only`: true
- `valid_for_casimir_input`: false
- `checks_ward_validation`: true
- `checks_unit_conversion`: false
- `checks_n0_policy`: false
- `production_use_allowed`: false
- 不允许 LSQ、response-level repair 或 quick audit 修正 production response。

## 复现入口

运行 `validation/outputs/response/bdg_finite_q/command.sh`。原始旧脚本输出会作为 ignored artifact 重新生成。

## 历史来源 / 旧 stage 对照

| 旧 stage 文件 | 现在对应的检验内容 | 当前状态 |
|---|---|---|
| `stageSC_1_bdg_finite_q_bare_kernel_audit.json` | 裸有限 q BdG kernel 检验 | 通过 |
| `stageSC_2b_bdg_amplitude_phase_gauge_restoration_audit.json` | 振幅/相位集体模 Schur 修正 Ward 检验 | 未通过 |
| `stageSC_2k_gauge_covariant_collective_package_audit.json` | gauge-covariant collective package control | 未通过 |
| `stageSC_3_bdg_normal_limit_audit.json` | `Delta -> 0` normal limit 检验 | 通过 |
| `stageSC_4_bdg_q0_limit_audit.json` | `q -> 0` local limit 检验 | 通过 |
| `stageSC_5_bdg_reflection_input_audit.json` | reflection input 候选接口检验 | 未通过 |
