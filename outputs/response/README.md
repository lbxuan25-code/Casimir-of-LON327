# response 输出入口

本目录当前只保留 finite-q response 主线的最新 active 结果：

- `finite_q_raw_q0_consistency/`

旧 finite-q 诊断、response 稳定性诊断、接口边界诊断已经移动到：

- `outputs/archive/response/`

阅读当前状态应先看：

1. `outputs/phase_reports/current_project_status.md`
2. `outputs/phase_reports/finite_q_response_status.md`
3. `outputs/response/finite_q_raw_q0_consistency/finite_q_raw_q0_consistency_summary.md`

旧数据仍然完整保留，可通过 `outputs/archive/ARCHIVE_INDEX.md` 追溯原路径和新路径。
这些旧数据不代表当前最终结论，也不能解释为正式 Casimir torque 结果。

维护原则：

- 不删除旧 outputs。
- 不使用 `.gitignore` 隐藏结果。
- 大型 `.csv`、`.npz`、`.png` 是复现数据，不是主要阅读入口。
