# TE/TM reflection input adapter 检验

## 检验目的

确认 dimensionless sheet conductivity 能否被格式化为 reflection input tensor，并进入 TE/TM reflection adapter 的候选路径；同时明确 prototype / scaffold 不等于 formal Casimir pipeline。

## 被检验对象

- reflection input tensor formatter；
- TE/TM reflection adapter；
- pre-Lifshitz readiness checks；
- trace-log integrand prototype；
- toy integration / material reflection grid scaffold；
- zero-mode grid convergence planning。

## 检验方法与判据

- 检查 reflection input tensor 是否有限、是否排除未定义的 q-zero 输入。
- 检查 TE/TM adapter formula deltas 和 q-sign consistency。
- 记录 prototype integrand、toy integration、material reflection grid 和 zero-mode grid planning 的 candidate 状态。
- 本检验不检查 upstream Ward closure，不接受 unit policy，不决定 `n=0` policy，不计算 formal energy、force 或 torque。

## 主要结果

### reflection input tensor formatting

状态：candidate / 诊断通过。

说明：formatter 可以把 dimensionless sheet conductivity 组织为 reflection input tensor，并保持有限输出。

### TE/TM reflection adapter

状态：candidate / 诊断通过。

说明：adapter 可以把 candidate tensor 放入 TE/TM reflection matrix 格式，并通过 adapter formula 与代表性 q-sign consistency 检查。

### pre-Lifshitz / integrand / toy integration prototype

状态：diagnostic-only。

说明：这些检查说明局部 prototype 路径可运行，但不计算正式 Lifshitz energy、force 或 torque。

### material reflection grid 与 zero-mode planning

状态：candidate / 仍需人工 policy 接受。

说明：latest material reflection grid prototype 记录为 passed candidate；zero-mode grid convergence audit 通过 planning 检查，但仍需接受 zero-mode 与 `Q -> 0` policy。

## 当前判定

candidate / diagnostic-only：reflection formatting 和 TE/TM adapter 有可复查证据，但不能绕过 upstream finite-q BdG、unit conversion 和 `n=0` policy gate。

## 对主流程的影响

- 不阻塞 local `q=0` response。
- 对 reflection input formatting 有支撑意义。
- 不允许 formal Casimir input。
- 不计算 energy、force 或 torque。

## 边界说明

- `diagnostic_only`: true
- `valid_for_casimir_input`: false
- `checks_ward_validation`: false
- `checks_unit_conversion`: depends on upstream unit conversion
- `checks_n0_policy`: planning only
- `production_use_allowed`: false

## 复现入口

运行 `validation/outputs/reflection/reflection_input/command.sh`。

## Source / status 对照

| source 文件 | 对应检验内容 | 当前状态 |
|---|---|---|
| `stage5_5b_reflection_input_tensor.json` | reflection input tensor formatting | candidate / 诊断通过 |
| `stage5_6_te_tm_reflection_adapter.json` | TE/TM reflection adapter | candidate / 诊断通过 |
| `stage5_7_pre_lifshitz_readiness_audit.json` | pre-Lifshitz readiness checks | diagnostic-only |
| `stage5_8_casimir_integrand_prototype.json` | trace-log integrand prototype | diagnostic-only |
| `stage5_10_toy_casimir_integration_convergence_audit.json` | toy integration convergence | diagnostic-only |
| `stage5_11c_real_material_reflection_grid_full36_order7_workers8.json` | material reflection grid prototype | candidate |
| `stage5_12_small_real_material_energy_prototype.json` | small material energy prototype | diagnostic-only |
| `stage5_13_zero_mode_grid_convergence_audit.json` | zero-mode grid convergence planning | candidate / policy pending |
