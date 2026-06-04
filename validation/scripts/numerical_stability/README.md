# 数值稳定性与响应诊断脚本

本目录保存数值稳定性、响应 contract 和前置物理边界诊断。多数脚本用于保留历史
可追溯性；当前 active diagnostic 包括：

- `diagnose_normal_finite_q_response.py`：第一阶段 normal-state finite-q
  current-current diagnostic；不属于完整 conductivity，也不接入 Casimir。

历史脚本用途包括：

- `audit_response_units.py`：早期单位链路审计；
- `diagnose_static_response.py`：早期静态响应边界诊断；

当前主计算入口仍位于 `scripts/`；本目录中的诊断均不得直接解释为主结果。
数值稳定性归纳请优先阅读
`docs/notes/numerical_stability_summary.md`，refined local-response 状态请优先阅读
`validation/outputs/archive/casimir/local_response_integral/refined_convergence/refined_convergence_summary.md`。
