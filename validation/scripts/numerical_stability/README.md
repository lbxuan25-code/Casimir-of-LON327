# 数值稳定性与响应诊断脚本

本目录保存数值稳定性、响应 contract 和前置物理边界诊断。多数脚本用于保留历史
可追溯性；当前 active diagnostic 包括：

- `diagnose_normal_finite_q_response.py`：第一阶段 normal-state finite-q
  current-current kernel convergence diagnostic；q=0 与 q!=0 都由同一 K 接口
  产生，只测试 n>=1 positive Matsubara 的 K(q)->K(0) same-interface 收敛。
  它不属于 gauge-closed finite-q conductivity，也不接入 Casimir。

历史脚本用途包括：

- `audit_response_units.py`：早期单位链路审计；
- `diagnose_static_response.py`：早期静态响应边界诊断；

当前主计算入口仍位于 `scripts/`；本目录中的诊断均不得直接解释为主结果。
数值稳定性归纳请优先阅读
`docs/notes/numerical_stability_summary.md`；finite-q Stage 1 状态请阅读
`validation/outputs/response/normal_finite_q_kernel_convergence/normal_finite_q_kernel_convergence_summary.md`。
