# Reflection input 摘要

## 检验对象

本目录归纳 reflection input tensor formatting、TE/TM reflection adapter、pre-Lifshitz readiness、Casimir integrand prototype、toy integration、material reflection grid 和 zero-mode grid convergence planning 的状态。

## 当前状态

- Stage 5.5b reflection input tensor formatter：`STAGE5_5B_REFLECTION_INPUT_FORMATTER_PASSED`。
- Stage 5.6 TE/TM adapter：`STAGE5_6_TE_TM_ADAPTER_PASSED`。
- Stage 5.7 pre-Lifshitz readiness：`STAGE5_7_PRE_LIFSHITZ_READINESS_PASSED`。
- Stage 5.8 integrand prototype：`STAGE5_8_CASIMIR_INTEGRAND_PROTOTYPE_PASSED`。
- Stage 5.10 toy integration：`STAGE5_10_TOY_CASIMIR_INTEGRATION_CONVERGENCE_AUDIT_PASSED`。
- Stage 5.11 material reflection grid 有早期 failed、monitor 和后续 passed 版本；latest consolidated status 记录 `STAGE5_11_REAL_MATERIAL_REFLECTION_GRID_PROTOTYPE_PASSED`。
- Stage 5.12 small real-material energy prototype：`STAGE5_12_SMALL_REAL_MATERIAL_ENERGY_PROTOTYPE_PASSED`，但仍不是物理预测。
- Stage 5.13 zero-mode grid convergence audit：`STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED`，仍需人工接受 zero-mode 与 Q->0 policy。

## 结论

reflection input formatting 和 TE/TM adapter 有候选通过证据，prototype/scaffold 路径可复查。但这不是完整 Lifshitz/Casimir production pipeline。

## 边界

raw response 没有 Ward validation、unit policy、`n=0` policy 时不能进入正式 Casimir input。本目录不改变任何数值状态，也不把 diagnostic-only 结果写成 production-ready。
