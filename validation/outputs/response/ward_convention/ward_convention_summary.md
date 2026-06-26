# Ward / response convention 数值检验

## 检验目的

确认 normal-state response 中 current vertex、contact term、density-current residual 和 left/right Ward source convention 是否一致，并记录 residual diagnostic 的适用边界。

## 被检验对象

- Peierls current vertex；
- finite-q contact term；
- density-current response residual；
- right Ward source convention；
- response-level residual diagnostic。

## 检验方法与判据

- 比较 bubble sign、direct contact bookkeeping 和 `C-K` 结构。
- 检查 right Ward source sign convention 对 residual 的影响。
- 对 corrected full response Ward residual 做数值闭合检查。
- 使用 targeted refinement 复查最坏参数 cluster。
- 本检验只报告 residual 和 convention，不修改 response，不拟合 contact，不使用 LSQ repair。

## 主要结果

### Peierls vertex 与 contact term convention

状态：诊断通过。

说明：positive bubble sign 与主路径 bookkeeping 得到支持，direct contact 与 `C-K` 结构的关系被记录为当前 convention 证据。

### density-current Ward residual convention

状态：诊断通过。

说明：right Ward diagnostic sign convention 已确认；旧 right residual 主要来自 diagnostic convention，而不是 response 公式需要被调参。

### corrected full response Ward residual

状态：诊断通过。

说明：corrected residual 的最大范数记录为 `4.139011615628368e-07`。targeted clean run 通过；但 user-run targeted refinement 仍提示部分 cluster 需要更高 refinement 或更宽 Fermi window。

### response-level repair / LSQ

状态：不适用 production。

说明：LSQ 或 response-level repair 只能作为 diagnostic reference，不能进入 production response pipeline。

## 当前判定

诊断通过：该目录支持 normal-state response convention，但不证明 superconducting finite-q gauge closure。

## 对主流程的影响

- 不阻塞 local response。
- 不直接提供 finite-q BdG response。
- 不提供 formal Casimir input。
- finite-q superconducting collective-sector closure 仍由 `validation/outputs/response/bdg_finite_q/` 控制。

## 边界说明

- `diagnostic_only`: true
- `valid_for_casimir_input`: false
- `checks_ward_validation`: true, but only for normal-state convention diagnostics
- `checks_unit_conversion`: false
- `checks_n0_policy`: false
- `production_use_allowed`: false

## 复现入口

运行 `validation/outputs/response/ward_convention/command.sh`。生成的旧脚本 JSON/MD/data/figures 是 ignored artifact。

## 历史来源 / 旧 stage 对照

| 旧 stage 文件 | 现在对应的检验内容 | 当前状态 |
|---|---|---|
| `stage4_13_bubble_sign_fix_regression.json` | bubble sign 与 direct contact bookkeeping | 诊断通过 |
| `stage4_17_right_ward_source_convention_audit.json` | right Ward source convention | 诊断通过 |
| `stage4_18_corrected_full_response_ward_validation.json` | corrected full response residual | 诊断通过 |
| `stage4_19_multi_parameter_ward_robustness_scan.json` | 多参数 robustness scan | 未完全闭合 |
| `stage4_20_targeted_no003_clean.json` | targeted refinement clean case | 诊断通过 |
