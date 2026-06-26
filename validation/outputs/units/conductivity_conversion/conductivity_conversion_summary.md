# Conductivity conversion 摘要

## 检验对象

本目录归纳 response-to-sheet-conductivity、bilayer sheet normalization、model units to SI sheet conductivity、dimensionless sheet conductivity 的诊断。

## 当前状态

- Stage 5.1 初始 convention audit：`CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE`，unit chain ambiguous。
- Stage 5.1b：`CONVENTION_FIXED`，bilayer sheet model convention 固定。
- Stage 5.2：`CONDUCTIVITY_SANITY_MONITOR_OFFDIAG`，属于 sanity / monitor。
- Stage 5.3：`CONDUCTIVITY_SYMMETRY_AUDIT_REQUIRES_FURTHER_SOURCE_SYMMETRY_AUDIT`，仍要求 source symmetry 复查。
- Stage 5.3b：`STAGE5_3B_PASSED_STABLE_FINITE_Q_LATTICE_TENSOR_EFFECT`。
- Stage 5.4a：`STAGE5_4A_CONDUCTIVITY_UNIT_CONVERSION_PASSED`。
- Stage 5.4b：`STAGE5_4B_CONDUCTIVITY_CONVERSION_PASSED`。

## 结论

单位链和 sheet conductivity formatting 已有候选通过证据，但这不是完整 response validation，也不允许 raw response 直接进入正式 Casimir input。

## 边界

本目录不验证 Ward closure，不处理 reflection adapter，不计算 energy、force 或 torque。当前结果是 production-relevant 的单位链证据，但整体仍为 diagnostic-only。
