# Casimir 主计算脚本

本目录保存已经形成初级结论的 local-response Casimir 主计算入口：

- `local_response_integral.py`：执行 local-response Matsubara、平行动量和角度积分。
- `local_response_distance_scan.py`：生成当前距离扫描初级结论与 zero-torque baseline。
- `local_response_config.py`：共享边界元数据与 torque tolerance。

当前计算边界固定为：

```text
local_response=True
finite_momentum_resolved=False
n0_policy=skip
benchmark_only=True
preliminary_local_response_conclusion=True
not_final_casimir_conclusion=True
```

收敛扫描、refinement 和流程诊断仍保存在 `validation/scripts/casimir/`。
