# q-grid mapping

本目录验证 q-grid / model-q mapping 的轻量证据。

## 本目录验证什么

- Casimir / reflection 积分中 model-q 和 physical-q 的覆盖关系；
- q-grid scaffold 是否提示 `Q=0` 与 response grid 覆盖限制；
- historical `casimir_q_grid_model_q_audit` 的结论入口。

## 本目录不验证什么

- 不验证 response kernel；
- 不验证 reflection matrix；
- 不验证正式 Lifshitz / Casimir 积分收敛；
- 不计算 energy、force 或 torque。

## production relevance

该目录只支持 grid planning 和单位/坐标映射复查。它不是 production Casimir evidence。

## diagnostic-only

当前状态为 diagnostic-only。任何 q-grid scaffold 通过都不能替代 Ward、unit、`n=0` policy gate。

摘要见 `q_grid_mapping_summary.md`，状态见 `q_grid_mapping_status.json`，复现入口见 `command.sh`。
