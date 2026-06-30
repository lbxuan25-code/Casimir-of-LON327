# finite-q BdG Casimir pipeline

本目录保存 finite-q BdG Casimir main production pipeline v1 的主结果输出。它不是 validation audit、smoke test 或草稿脚本；validation 目录只用于后续审查 Ward residual、`n=0` 稳定性、grid 收敛性和 benchmark。

## 主脚本

`scripts/casimir/finite_q_bdg_casimir_pipeline.py`

## Response contract

- pairing 输入：`PairingAnsatz`
- engine：`finite_q_bdg_response_from_ansatz`
- `phase_vertex = "bond_endpoint_gauge"`
- `current_vertex = "peierls"`
- `collective_mode = "amplitude_phase"`
- `collective_counterterm = "goldstone_gap_equation"`
- `include_phase_phase_direct = true`
- Casimir input 固定使用 `full_response = amplitude_phase_schur`
- `bare_bubble`、`direct`、`bare_total`、`minus_schur`、`plus_schur` 只作为 diagnostic components，不参与 response selection

Ward residual 作为每个计算点的 numerical quality metadata 记录，不作为本轮 pipeline 的阻断条件。

## 任务结构与距离复用

主流程已经拆成 heavy reflection layer 和 cheap energy layer：

- `reflection_results.shard_*_of_*.jsonl`：单片 plate reflection cache。key 只包含 `pairing`、`plate_theta_deg`、`n`、`Q_index`、`phi_index`、`n0_policy` 和 `config_hash`；`distance_nm` 不在 heavy key 中。
- `energy_point_results.shard_*_of_*.jsonl`：由缓存的左片 `theta=0` 和右片 `theta` 组合得到 roundtrip 后，对所有距离展开，只计算 `exp(-2*kappa*d)`、`log det` 和积分贡献。

因此在相同 pairing、angle、Matsubara、Q、phi grid 下，增加距离只增加 cheap energy points，不增加 finite-q BdG response / reflection 任务。`--dry-run` 会分别报告 `num_plate_reflection_tasks`、`num_roundtrip_tasks` 和 `num_energy_points`，并标记 `distance_reuse_enabled=true`。

## Unit contract

主流程调用 `spatial_response_to_bilayer_sheet_conductivity_model`、`model_response_to_sheet_conductivity` 和 `sheet_conductivity_to_reflection_dimensionless`。

```text
Pi_ij = full_response[1:3, 1:3]
sigma_model = - Pi_ij / omega_eV
sigma_sheet_SI = (e^2 / hbar) * sigma_model
sigma_tilde = sigma_sheet_SI / sigma0
```

当前材料边界为 `bilayer_normalized_2D_sheet`，不是 bulk 3D，也不是 finite-thickness slab。

## Reflection contract

主流程调用 `sigma_tilde_xy_to_te_tm_reflection_matrix`。conductivity rotation 发生在 xy basis，然后进入 xy -> LT -> TE/TM adapter。

## Casimir contract

```text
F(d)/A = k_B T sum_n' int Q dQ dphi/(2pi)^2
         log det[I - exp(-2*kappa*d) R1 @ R2]
```

`logdet_real` 进入 energy，`logdet_imag` 作为数值误差 metadata。

trace-log 单点 helper 来自 `src/lno327/lifshitz_integrand.py`，语义为 main pipeline 的外部 Matsubara/Q grid summation 组件；它本身不声称完成 full integral。

## Quadrature contract

默认 `--integration-strategy best_available_adaptive` 调用 `src/lno327/finite_q_quadrature.py`：

- `finite_q_quadrature_points(q_model, options)` 返回 points、weights 和 metadata；
- adaptive mesh 使用当前 plate crystal frame 中的 `q_crystal = R(-theta) q_lab`；
- refined parent cell 被 Gauss subcell nodes 替代，parent 与 children 不 double-count；
- 每个 reflection row 记录 `num_quadrature_points`、`num_cells_total`、`num_cells_refined` 和 `quadrature` metadata。

如果选择 `--integration-strategy uniform`，summary/status 会写明 uniform midpoint mesh，并将 adaptive 参数记录为 `null` 或 disabled。

## n0 policy

默认 `--n0-policy extrapolate`：`n=0` 不直接使用 `omega_eV=0` 做 `sigma=-K/omega`，而使用低频 reflection matrix 外推。每个 n=0 reflection row 记录：

- `n0_extrapolation_method = linear_in_omega_eV`
- `n0_extrapolation_omega_eV`
- `n0_extrapolation_order`
- `n0_reflection_norms`
- `n0_reflection_norm_variation`
- `n0_fit_residual_norm`
- `n0_stability_status`

也支持 `--n0-policy skip`；skip 时 summary/status 会记录 `complete_matsubara_sum=false`、`complete_except_n0=true`、`not_final_full_matsubara_result=true`，不应声称 full Matsubara production result。

## 输出文件

- `run_config.json`：标准化配置和 `config_hash`；
- `reflection_results.shard_*_of_*.jsonl`：active heavy reflection shard 输出；
- `energy_point_results.shard_*_of_*.jsonl`：active cheap energy shard 输出；
- `failed_points.shard_*_of_*.jsonl`：active shard 失败点和 traceback；
- `reflection_results.jsonl` / `point_results.jsonl` / `failed_points.jsonl`：finalize 或 plot-only 合并产物；
- `summary.json` / `status.json`：pipeline contract、grid、质量 metadata 和输出清单；
- `run_status.json`：长任务运行状态；
- `logs/run.log` / `logs/errors.log`：运行日志；
- `data/*.csv`：主图数据源；
- `figures/*.png`：论文主图风格结果图。

## 服务器运行

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

bash outputs/casimir/finite_q_bdg_pipeline/command.sh
```

后台运行：

```bash
nohup bash outputs/casimir/finite_q_bdg_pipeline/command.sh \
  > outputs/casimir/finite_q_bdg_pipeline/logs/nohup.log 2>&1 &
```

查看进度：

```bash
tail -f outputs/casimir/finite_q_bdg_pipeline/logs/nohup.log
tail -f outputs/casimir/finite_q_bdg_pipeline/logs/run.log
```

## 断点续跑

默认命令带 `--resume`。重新运行同一 grid 时，脚本会读取当前 shard 的 `reflection_results.shard_*_of_*.jsonl`，跳过已完成的 plate reflection task key。

`run_config.json` 保存 `config_hash`。如果当前配置和已有输出目录中的 hash 不一致，`--resume` 默认报错，避免 coarse grid、temperature、eta、q grid 或 integration strategy 改变后误复用旧结果。只有明确确认时才使用 `--allow-config-mismatch`。

## plot-only

```bash
python scripts/casimir/finite_q_bdg_casimir_pipeline.py \
  --plot-only \
  --output-dir outputs/casimir/finite_q_bdg_pipeline
```

该模式读取所有 shard-specific reflection / energy / failed JSONL，合并为兼容文件，并重建 `data/*.csv`、`figures/*.png`、`summary.json`。

## 分片运行

```bash
python scripts/casimir/finite_q_bdg_casimir_pipeline.py \
  --task-shard-index 0 \
  --task-shard-count 4 \
  --resume \
  --output-dir outputs/casimir/finite_q_bdg_pipeline
```

分片规则为 `task_index % task_shard_count == task_shard_index`。

各 shard 写入独立 active JSONL，不会多个进程同时写同一个 `point_results.jsonl`。合并文件只在 finalize / plot-only 阶段生成。

## 图像开关

`--distance-scan`、`--angle-scan`、`--heatmap-scan`、`--pairing-comparison` 控制对应图像是否生成；CSV 网格和 summary 仍按已完成 energy rows 重建。

## 运行边界

Codex 不应启动本目录的正式服务器 full run；正式 `command.sh` 由用户手动执行和审查。

## 当前科学边界

本目录按正式主计算工程标准产出数据和图像，但 `summary.json` / `status.json` 明确保留：

```text
valid_for_formal_casimir_claim = false
not_final_material_conclusion = true
ward_residual_recorded_not_gating = true
```

这些结果可用于论文图像初稿和主线数值包，不应被表述为无边界最终材料结论。
