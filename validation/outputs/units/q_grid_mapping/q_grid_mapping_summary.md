# q-grid 到 model-q 映射检验

## 检验目的

确认 Casimir / reflection planning 中 physical-q 到 model-q 的转换范围、small-q diagnostic 覆盖范围，以及 production grid 需要注意的 `Q=0` 和 response coverage 限制。

## 被检验对象

- physical-q 到 model-q 的单位换算；
- integration q-grid scaffold；
- small-q diagnostic list；
- Casimir-relevant q range；
- `Q=0` TE/TM direction warning。

## 检验方法与判据

- 扫描不同 in-plane conversion length 与 separation 下的 `q_model` 范围。
- 比较 small-q diagnostic list 与 Casimir-relevant q range。
- 检查 grid scaffold 是否保留 `Q=0` 和 response-grid insufficiency warning。
- 本检验不计算 response tensor，不产生 finite-q conductivity，不验证 reflection adapter，不给出 Casimir 结论。

## 主要结果

### q-grid 到 model-q 映射

状态：诊断通过。

说明：historical unit/sampling audit 记录 full-grid `q_model_max = 1.05333`，`q_model_max/pi = 0.335286`。

### small-q diagnostic 覆盖范围

状态：未完成 production 覆盖。

说明：normal finite-q kernel diagnostic 的 sampled q range 最大为 `0.005`，只覆盖 small-q limit，不覆盖当前 Casimir-relevant model-q range。

### q-grid scaffold warning

状态：诊断通过。

说明：scaffold 明确记录 `Q=0` 的 TE/TM in-plane direction 需要 symmetry/limit 处理或从 angular-grid production runs 排除；也明确记录既有 8 个 validation reflection cases 不是 production integration grid。

## 当前判定

诊断通过：该目录支持 q-grid planning，但 production integration grid 尚未由此完成。

## 对主流程的影响

- 不阻塞 local response。
- 不验证 finite-q BdG response。
- 不验证 reflection input。
- 不允许 formal Casimir input。

## 边界说明

- `diagnostic_only`: true
- `valid_for_casimir_input`: false
- `checks_ward_validation`: false
- `checks_unit_conversion`: true, only for q mapping scale
- `checks_n0_policy`: false
- `production_use_allowed`: false

## 复现入口

运行 `validation/outputs/units/q_grid_mapping/command.sh`。

## 历史来源 / 旧 stage 对照

| 旧 stage 文件 | 现在对应的检验内容 | 当前状态 |
|---|---|---|
| `casimir_q_grid_model_q_audit_summary.md` | physical-q 到 model-q unit/sampling audit | 诊断通过 |
| `stage5_9_casimir_grid_planning_scaffold.json` | q-grid scaffold warning 与 planning | 诊断通过 |
