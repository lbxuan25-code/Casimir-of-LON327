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

finite-q BdG 主流程采用两层结果结构：

- heavy layer：`reflection_results.shard_*_of_*.jsonl`，按 `pairing / plate_theta / n / Q / phi / n0_policy / config_hash` 计算单片反射。`distance_nm` 不进入 heavy task key，因此距离扫描不会重算 finite-q BdG response 或 reflection。
- cheap layer：`energy_point_results.shard_*_of_*.jsonl`，由缓存的左片 `theta=0` 和右片 `theta` 反射矩阵展开所有距离，只计算 `exp(-2*kappa*d)`、trace-log 和积分权重。

`--plot-only` 会合并 shard-specific JSONL，重建兼容用的 `reflection_results.jsonl`、`point_results.jsonl`、`failed_points.jsonl`、`data/*.csv` 和 `figures/*.png`。

`best_available_adaptive` 不再是 metadata 占位符；主流程调用 `src/lno327/workflows/finite_q_quadrature.py` 中的 q-specific adaptive quadrature。若使用 `--integration-strategy uniform`，summary 会明确记录 uniform mesh，并关闭 adaptive metadata。

断点续跑使用 `run_config.json` 中的 `config_hash`。修改温度、gap、eta、q grid、Matsubara grid、integration strategy 或 contract 后，`--resume` 默认拒绝复用旧输出；只有显式传入 `--allow-config-mismatch` 才允许继续。

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
