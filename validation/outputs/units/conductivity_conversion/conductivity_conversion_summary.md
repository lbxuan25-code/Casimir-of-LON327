# Response 到 sheet conductivity 的单位转换检验

## 检验目的

确认当前 model response 到 bilayer sheet conductivity、SI sheet conductivity 和 dimensionless sheet conductivity 的候选单位链是否自洽。

## 被检验对象

- response-to-sheet-conductivity convention；
- bilayer sheet model normalization；
- model conductivity 到 SI sheet conductivity 的单位转换；
- SI sheet conductivity 到 dimensionless sheet conductivity 的转换。

## 检验方法与判据

- 比较 model response 与 bilayer sheet convention 的相对结构。
- 检查单位转换后 diagonal / offdiagonal 结构是否保持有限且相对结构一致。
- 记录 source symmetry / offdiag monitor 的状态。
- 本检验不检查 Ward closure，不检查 reflection matrix，不处理 `n=0` policy，不计算 Casimir。

## 主要结果

### response 到 sheet conductivity 的转换

状态：candidate。

说明：初始 convention audit 显示仅从代码路径不能唯一确定 convention；后续 bilayer sheet model convention 已固定为当前候选。

### conductivity sanity 与 source symmetry

状态：未完成。

说明：conductivity sanity scan 属于 offdiag monitor；symmetry / convergence audit 仍要求进一步 source symmetry 复查。

### SI / dimensionless sheet conductivity

状态：诊断通过 / candidate。

说明：unit conversion 和 dimensionless sheet conductivity conversion 给出通过证据，且 conversion preserves relative structure。但这只说明单位链候选可用，不代表 upstream response 已可作为 formal Casimir input。

## 当前判定

诊断通过 / candidate：单位链有可复查证据，但整体仍依赖 upstream Ward validation、unit policy 和 `n=0` policy。

## 对主流程的影响

- 不阻塞 local `q=0` response。
- 对 unit conversion 具有支撑意义。
- 不单独允许 reflection input 或 formal Casimir input。

## 边界说明

- `diagnostic_only`: true
- `valid_for_casimir_input`: false
- `checks_ward_validation`: false
- `checks_unit_conversion`: true
- `checks_n0_policy`: false
- `production_use_allowed`: false

## 复现入口

运行 `validation/outputs/units/conductivity_conversion/command.sh`。

## 历史来源 / 旧 stage 对照

| 旧 stage 文件 | 现在对应的检验内容 | 当前状态 |
|---|---|---|
| `stage5_1_response_to_conductivity_convention_audit.json` | response-to-conductivity convention 是否唯一 | 未完成 |
| `stage5_1b_bilayer_sheet_conductivity_convention.json` | bilayer sheet model convention | candidate |
| `stage5_2_bilayer_sheet_conductivity_sanity_scan.json` | conductivity sanity / offdiag monitor | diagnostic monitor |
| `stage5_3_bilayer_sheet_conductivity_symmetry_convergence_audit.json` | source symmetry / convergence audit | 未完成 |
| `stage5_3b_bilayer_sheet_conductivity_offdiag_convergence_audit.json` | finite-q lattice tensor offdiag convergence | 诊断通过 |
| `stage5_4a_conductivity_unit_conversion.json` | SI unit conversion | 诊断通过 |
| `stage5_4b_si_sheet_dimensionless_conductivity.json` | dimensionless sheet conductivity | 诊断通过 |
