# 数值稳定性历史诊断脚本归档

本目录保存已经完成阶段性任务的历史诊断入口。它们不删除，是为了保留可追溯性；
但它们不是当前推荐的主入口，也不应用作新的正式 benchmark 起点。

当前推荐入口仍在 `scripts/` 顶层，至少包括：

- `validation/scripts/numerical_stability/benchmark_normal_fs_adaptive_integration.py`
- `scripts/casimir/local_response_integral.py`
- `validation/scripts/casimir/converge_casimir_local_response_integral.py`
- `validation/scripts/casimir/refine_casimir_local_convergence_blockers.py`
- `validation/scripts/casimir/run_casimir_local_convergence_final.py`

本目录中的脚本用途：

- `audit_response_units.py`：早期单位链路审计；
- `diagnose_static_response.py`：早期静态响应边界诊断；

这些脚本只应在追溯旧阶段判断时使用。新的数值稳定性归纳请优先阅读
`docs/notes/numerical_stability_summary.md`，refined local-response 状态请优先阅读
`validation/outputs/archive/casimir/local_response_integral/refined_convergence/refined_convergence_summary.md`。
