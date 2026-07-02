# finite-q BdG Casimir smoke-pilot output

本目录归档一次 finite-q BdG Casimir pipeline 的 smoke-pilot 轻量输出。参数级别用于流程检查和图像管线预览，不是收敛 production 计算。

边界说明：

- `diagnostic_only = true`
- `valid_for_formal_casimir_claim = false`
- Ward identity not closed / Ward residual recorded，不作为本轮 smoke-pilot 的 gating 条件
- 原始 11 张 pipeline 图保留在 `figures/`
- `figures/normal_subtracted/` 中的三张图由 `scripts/casimir/postprocess_normal_subtracted_figures.py` 从原始 CSV 纯后处理派生生成
- 本目录归档不包含 logs、JSONL、raw/intermediate/cache 或 active runtime artifacts

可追踪的轻量证据包括 `run_config.json`、`summary.json`、`status.json`、`run_status.json`、`data/*.csv`、`figures/*.png` 和 `figures/normal_subtracted/README.md`。
