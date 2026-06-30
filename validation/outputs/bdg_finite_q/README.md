# BdG finite-q 验证输出

本目录保存 diagnostic-only finite-q BdG validation 的轻量输出和复现入口。这里的结果用于定位 superconducting BdG finite-q Ward closure blocker，不是 formal finite-q Casimir 输入。

`bdg_finite_q_validation_status.json` 是当前唯一 active BdG finite-q validation status marker。旧目录 `validation/outputs/response/bdg_finite_q/` 只保留迁移说明，不再保存 active status JSON。

当前 q=0 precondition 状态由统一脚本 `validation/scripts/bdg_finite_q/q0_bdg_response_alignment.py` 生成：`spm: convention_aware_pass`，`dwave: intraband_aware_pass`，`normal/onsite_s: diagnostic_only_not_passed`。`dwave` raw-vs-total q=0 mismatch 仍在报告中可见，但由 local intraband / `-f'(E)` 贡献解释，不再作为未解释 raw-bubble/vertex mismatch。

finite-q Ward scan 会单独报告 `diagnostic_run_completed` 与 `ward_identity_closed`。当前 finite-q Ward closure 仍未完成，所有 finite-q validation 输出都保持 `valid_for_casimir_input=False`。

## 可提交内容

- 小型 markdown summary；
- 小型 JSON / CSV status 或 summary；
- 复现命令文件；
- 简短人工阅读报告。

## 不应提交内容

- raw arrays；
- 大型 CSV；
- expanded logs；
- cache tensors；
- 可重复生成的中间响应矩阵；
- formal Casimir 输入或 torque 结论。

所有 generated finite-q validation results 都必须保持 `valid_for_casimir_input=False`。若需要复查大型数据，请运行本目录的 `command.sh` 或 `validation/scripts/bdg_finite_q/` 下的脚本，在本地重新生成。
