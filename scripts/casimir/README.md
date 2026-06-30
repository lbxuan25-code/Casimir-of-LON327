# Casimir 主计算脚本

本目录保存 Casimir 主计算入口：

- `finite_q_bdg_casimir_pipeline.py`：finite-q BdG Casimir main production pipeline v1，串联 response、unit、reflection、trace-log 和主图输出。
- `local_response_integral.py`：执行 local-response Matsubara、平行动量和角度积分。
- `local_response_distance_scan.py`：生成当前距离扫描初级结论与 zero-torque baseline。
- `local_response_config.py`：共享边界元数据与 torque tolerance。

finite-q BdG 主流程的当前边界为：

```text
main_production_pipeline_v1=True
full_response_source=amplitude_phase_schur
ward_residual_recorded_not_gating=True
valid_for_formal_casimir_claim=False
not_final_material_conclusion=True
```

local-response baseline 的边界固定为：

```text
local_response=True
finite_momentum_resolved=False
n0_policy=skip
benchmark_only=True
preliminary_local_response_conclusion=True
not_final_casimir_conclusion=True
```

收敛扫描、refinement 和流程诊断仍保存在 `validation/scripts/casimir/`。
