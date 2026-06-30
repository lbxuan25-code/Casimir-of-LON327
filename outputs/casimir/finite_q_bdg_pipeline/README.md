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

## n0 policy

默认 `--n0-policy extrapolate`：`n=0` 不直接使用 `omega_eV=0` 做 `sigma=-K/omega`，而使用低频 reflection matrix 外推。也支持 `--n0-policy skip`；skip 时 summary/status 会记录 `complete_matsubara_sum=false` 语义，不应声称 full Matsubara production result。

## 输出文件

- `point_results.jsonl`：每个 deterministic task 的结果，完成一个点立即 append；
- `failed_points.jsonl`：失败点和 traceback；
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

默认命令带 `--resume`。重新运行同一 grid 时，脚本会读取 `point_results.jsonl`，跳过已完成的 deterministic task key。

## plot-only

```bash
python scripts/casimir/finite_q_bdg_casimir_pipeline.py \
  --plot-only \
  --output-dir outputs/casimir/finite_q_bdg_pipeline
```

该模式只读取已有 `point_results.jsonl` 并重建 `data/*.csv`、`figures/*.png`、`summary.json`。

## 分片运行

```bash
python scripts/casimir/finite_q_bdg_casimir_pipeline.py \
  --task-shard-index 0 \
  --task-shard-count 4 \
  --resume \
  --output-dir outputs/casimir/finite_q_bdg_pipeline
```

分片规则为 `task_index % task_shard_count == task_shard_index`。

## 当前科学边界

本目录按正式主计算工程标准产出数据和图像，但 `summary.json` / `status.json` 明确保留：

```text
valid_for_formal_casimir_claim = false
not_final_material_conclusion = true
ward_residual_recorded_not_gating = true
```

这些结果可用于论文图像初稿和主线数值包，不应被表述为无边界最终材料结论。
