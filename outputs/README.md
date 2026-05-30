# Outputs Guide

本目录只放可复现数据产物、运行缓存和历史归档。阶段报告已经移到
`docs/reports/`，避免把“结论入口”和“大型输出文件”混在一起。

## active outputs

active outputs 只保留当前主线或最新 benchmark：

- `response/finite_q_raw_q0_consistency/`：当前 finite-q 主线最新诊断。
- `casimir/local_response_integral/distance_scan/`：local-response distance scan baseline。
- `cache/casimir_local_response/response_tensors/`：distance scan 仍引用的 response cache。
- `numerical_stability/`：数值稳定性阶段入口说明。

## archive outputs

历史诊断结果已移动到：

- `archive/response/`
- `archive/normal_state/`
- `archive/casimir/`
- `archive/smoke/`

archive 保存历史诊断结果和中间 benchmark，不作为当前阅读入口。需要追溯旧路径时，
先看 `archive/ARCHIVE_INDEX.md`。

## 阅读顺序

1. `../docs/reports/current_project_status.md`
2. `../docs/reports/finite_q_response_status.md`
3. `../docs/reports/local_response_baseline_status.md`
4. `response/finite_q_raw_q0_consistency/finite_q_raw_q0_consistency_summary.md`

## 维护原则

- 不删除历史数据。
- 不使用或修改 `.gitignore` 隐藏结果。
- 大型 `.csv`、`.npz`、`.png` 是复现数据，不是主要阅读入口。
- active 和 archive 输出都不是正式 Casimir torque 结论。
