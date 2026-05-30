# finite-q local-limit 分解诊断输出

本目录保存 finite-q response local-limit decomposition diagnostic 的输出。该阶段只回答：
当前 finite-q bubble 在 `q -> 0` 时更接近哪一个 local response component。

比较对象包括：

- `local_sigma`
- `local_K_para`
- `local_K_total`
- `local_K_total_over_omega`
- `normal_kubo_sigma`

不适用于某个 kind 的 component 会写为 NaN，并在 notes 中说明。本目录不是 Casimir
结果目录，不接入 Lifshitz 积分，也不输出 torque 结论。

当前限制：

- `gauge_status=prototype_not_ward_verified`
- Ward identity / diamagnetic closure 未完成
- n=0 model 未完成
- `final_casimir_input=False`
- `not_final_Casimir_conclusion=True`

主要文件：

- `data/finite_q_local_limit.csv`
- `data/finite_q_local_limit.npz`
- `finite_q_local_limit_summary.md`
- `figures/small_q_error_vs_q.png`
- `figures/small_q_error_vs_nk.png`
- `figures/best_local_component_match.png`
- `figures/component_error_heatmap.png`
