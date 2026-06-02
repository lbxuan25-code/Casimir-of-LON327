# finite-q subspace / denominator repair 诊断摘要

本轮目标是 subspace / denominator repair diagnostic：处理 small-q continuity 中
near-degenerate band/subspace 的 overlap、band phase、band order，以及 denominator 数值稳定性。

上一轮已经看到 vertex_mismatch_detected=False 且 BZ wrapping 未触发，因此本轮不优先改 vertex 或 BZ wrapping。
本轮仍只做 response 层 quick 诊断，不接入 Lifshitz/Casimir，也不输出 torque 结论。

kinds=['normal', 'spm', 'dwave']
matsubara_list=[1]
q_list=[0.0001, 0.0005, 0.001, 0.005]
q_phi_list=[0.0, 0.7853981634]
nk_list=[6, 8]
deg_tol_list=[1e-08, 1e-07, 1e-06]
denominator_mode_list=['raw', 'stable']

max_eigenstate_overlap_offdiag_norm=0.101588
max_projector_overlap_error=0.00257512
projector_overlap_smaller_than_eigenstate_offdiag=True
possible_true_subspace_mixing=False
max_near_degenerate_count=64
max_denominator_regularization_delta=3.02636e-16
raw_best_small_q_relative_error=3.15531e-09
stable_best_small_q_relative_error=3.15531e-09
stable_min_q_relative_error=3.15531e-09
stable_denominator_improves_continuity=False
small_q_error_decreases_toward_q0=True
deg_tol_conclusion_stable=True
A4_q_to_zero_trend_tested=False
recommend_return_to_finite_q_A4_anisotropy_diagnostic=False
recommend_continue_formula_repair=True
recommend_subspace_safe_response_prototype=False
recommend_denominator_stable_mode_refinement=False

## 限制
- gauge_status=prototype_not_ward_verified
- Ward identity / diamagnetic closure 未完成
- n=0 model 未完成
- final_casimir_input=False
- not_final_Casimir_conclusion=True
