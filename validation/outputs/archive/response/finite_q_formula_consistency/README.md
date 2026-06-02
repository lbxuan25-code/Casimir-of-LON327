# finite-q formula / vertex consistency 诊断输出

本目录保存 finite-q response formula / vertex consistency diagnostic 的输出。该阶段只检查
finite-q prototype 在 small-q 下的不平滑来源，包括 vertex、band overlap、BZ wrapping、
denominator stability 和 local component error。

本目录不是 Casimir 结果目录，不接入 Lifshitz 积分，也不输出 torque 结论。

当前限制：

- `gauge_status=prototype_not_ward_verified`
- Ward identity / diamagnetic closure 未完成
- n=0 model 未完成
- `final_casimir_input=False`
- `not_final_Casimir_conclusion=True`

主要文件：

- `data/finite_q_formula_consistency.csv`
- `data/finite_q_formula_consistency.npz`
- `finite_q_formula_consistency_summary.md`
- `figures/small_q_error_vs_q.png`
- `figures/vertex_error_vs_q.png`
- `figures/overlap_error_vs_q.png`
- `figures/component_error_comparison.png`
