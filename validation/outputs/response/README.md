# Response validation 轻量证据目录

本目录只保存 response 层级的长期验证证据：中文 README、summary、status marker 和复现入口。它不是 raw numerical artifacts 仓库。

`data/`、`figures/`、`raw/`、`intermediate/` 默认视为可再生成输出，并由 `.gitignore` 排除。

## 当前子目录

1. `normal_finite_q_kernel/`：验证 normal-state finite-q current-current kernel 的同接口与收敛行为。
2. `bdg_finite_q/`：验证 finite-q BdG superconducting response 的 diagnostic 状态和 Casimir input gate。
3. `ward_convention/`：验证 normal-state Ward / response convention。

单位换算、q-grid mapping、reflection input 已拆出 response 层级：

- `validation/outputs/units/conductivity_conversion/`
- `validation/outputs/units/q_grid_mapping/`
- `validation/outputs/reflection/reflection_input/`

## 本目录验证什么

- response kernel / response convention 的数值一致性；
- finite-q BdG pipeline 是否具备通过 Casimir input gate 的条件；
- Ward diagnostic convention 是否支持当前 response 约定。

## 本目录不验证什么

- 不验证 SI 单位换算链；
- 不验证 q-grid 与 model-q mapping；
- 不验证 TE/TM reflection adapter；
- 不计算正式 Casimir energy、force 或 torque。

## production relevance

`normal_finite_q_kernel/` 和 `ward_convention/` 是 response convention 与 kernel 层的支撑证据。`bdg_finite_q/` 是 production gate：当前 consolidated status 为 `FAILED`，因此 finite-q BdG response 仍不能作为正式 Casimir input。

## diagnostic-only 边界

raw finite-q BdG response、LSQ/response-level repair、quick audit 和任何未通过 consolidated gate 的结果均为 diagnostic-only。
