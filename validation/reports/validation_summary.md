# Validation 总览

本总览按具体数值检验内容组织，不以历史 stage 编号作为阅读入口。详细结论见各 `validation/outputs/**` 子目录。

## 数值验证状态总览

| 模块 | 具体检验 | 当前状态 | 是否阻塞主流程 | 边界 |
|---|---|---|---|---|
| normal finite-q response | current-current kernel 收敛与 `q -> 0` 同接口一致性 | 诊断通过 | 不阻塞 local response | 不是 gauge-closed conductivity |
| finite-q BdG response | 裸 kernel、集体模 Schur/Ward、`Delta -> 0` normal limit、`q -> 0` local limit | 部分通过；集体模 Ward restoration 未通过 | 阻塞 formal finite-q Casimir input | raw response 是 diagnostic-only |
| Ward convention | Peierls/contact convention、density-current residual convention、right Ward source convention | 诊断通过 | 不直接阻塞 | 只报告 convention，不修 response |
| unit conversion | response 到 sheet conductivity、SI / dimensionless conductivity | candidate / 诊断通过 | 依赖 upstream response | 不单独证明 Casimir-ready |
| q-grid mapping | model-q 覆盖范围、`Q=0` warning、production grid 覆盖提醒 | 诊断通过 | 不直接阻塞 | 不验证 response 或 reflection |
| reflection input | reflection tensor formatting、TE/TM adapter、prototype grid | candidate / 诊断通过 | 依赖 upstream validation | 不计算 energy/force/torque |
| local-response Casimir benchmark | local-response integration 和 cache / convergence scaffold | diagnostic benchmark | 不等价于 finite-q production | 需新鲜 summary 支撑正式结论 |
| smoke / plumbing | 脚本入口、cache、最小计算路径 | operational pass/fail | 失败会阻塞开发信心 | 不定义物理结论 |

## Normal finite-q current-current kernel

检验目的：确认 normal-state finite-q current-current kernel 在 `q=0` 与 `q!=0` 下使用同一接口，并检查小 q 数值连续性。

当前结论：`q=0` 同接口误差为 `0`，最小非零 q 为 `0.0001`，该点最大 same-interface error 约为 `2.05641e-4`，C4 covariance error 约为 `6.58e-15`，所有 kernel 分量有限。

当前判定：诊断通过。它支撑 kernel 层接口一致性，但不是完整 finite-q conductivity，不包含 `n=0` true static，也不作为 Casimir input。

复现入口：`validation/outputs/response/normal_finite_q_kernel/command.sh`。

## finite-q BdG superconducting response

检验目的：确认 finite-q BdG response 是否可作为下游 formal Casimir input。

当前结论：裸有限 q BdG kernel、`Delta -> 0` normal limit 和 `q -> 0` local limit 通过诊断；振幅/相位集体模 Schur/Ward restoration 未通过；reflection input candidate 因 upstream validation 未通过而不能正式使用。

当前判定：未通过。它不阻塞 local `q=0` response，但阻塞 raw finite-q BdG response 进入 formal Casimir input。

复现入口：`validation/outputs/bdg_finite_q/command.sh`。

## Ward / response convention

检验目的：确认 normal-state response 中 Peierls current vertex、contact term、density-current residual 和 right Ward source convention 的一致性。

当前结论：positive bubble sign、right Ward diagnostic sign convention 和 corrected full-response Ward residual 获得数值支持；targeted refinement clean case 通过，但部分 user-run cluster 仍提示需要更高 refinement 或更宽 Fermi window。

当前判定：诊断通过。它支持 response convention，但不修改 response，也不证明 superconducting finite-q gauge closure。

复现入口：`validation/outputs/response/ward_convention/command.sh`。

## Unit conversion

检验目的：确认 model response 到 bilayer sheet conductivity，再到 SI sheet / dimensionless sheet conductivity 的单位链候选。

当前结论：初始 convention audit 指出 unit chain ambiguous；后续 bilayer sheet convention、unit conversion 和 dimensionless sheet conductivity conversion 给出 candidate 通过证据；conductivity symmetry 仍要求 source symmetry 复查。

当前判定：诊断通过 / candidate。它 production-relevant，但依赖 upstream Ward validation、unit policy 和 `n=0` policy。

复现入口：`validation/outputs/units/conductivity_conversion/command.sh`。

## q-grid mapping

检验目的：确认 Casimir / reflection planning 中 physical-q 到 model-q 的覆盖范围和 warning。

当前结论：grid scaffold 记录了 `Q=0` TE/TM 方向问题和 response grid 覆盖不足；historical q-grid audit 显示 small-q diagnostic list 只覆盖 `q_model <= 0.005`，不覆盖当前 Casimir-relevant model-q range。

当前判定：诊断通过。它只支持 grid planning，不验证 response、reflection adapter 或 full integration。

复现入口：`validation/outputs/units/q_grid_mapping/command.sh`。

## Reflection input

检验目的：确认 dimensionless sheet conductivity 能否被格式化为 reflection input，并检查 TE/TM adapter 的候选 convention。

当前结论：reflection input tensor formatter 和 TE/TM adapter 给出 candidate 通过证据；prototype/scaffold 路径可运行；但 raw response 没有通过 upstream Ward/unit/`n=0` gate 时不能进入 formal Casimir。

当前判定：candidate / diagnostic-only。它不计算正式 energy、force 或 torque。

复现入口：`validation/outputs/reflection/reflection_input/command.sh`。

## 历史来源说明

旧 stage 输出已合并到对应 summary 和 status JSON。stage 名称不再作为阅读入口；需要追溯脚本时，请查看各 summary 的“历史来源 / 旧 stage 对照”或对应 `command.sh`。
