# finite-q formula / vertex consistency 诊断摘要

本轮目标是 formula / vertex consistency repair 诊断：检查 finite-q response 在 small-q 下
不平滑的来源，而不是扩大扫描、接入 Lifshitz/Casimir 或输出 torque 结论。

当前 finite-q response 仍不是 Ward 完备。
kinds=['normal', 'spm', 'dwave']
matsubara_list=[1]
q_list=[0.0001, 0.0005, 0.001, 0.005]
q_phi_list=[0.0, 0.7853981634]
nk_list=[6, 8]
temperature=30.0
delta0=0.04
eta=0.0001

vertex_mismatch_detected=False
max_vertex_relative_error=0
overlap_or_band_phase_issue_detected=True
max_overlap_diagonal_error=0.00128839
max_overlap_offdiag_norm=0.101588
bz_wrapping_issue_detected=False
max_wrapped_fraction=0
denominator_instability_detected=True
max_near_degenerate_count=24
best_small_q_relative_error=3.15531e-09
small_q_continuity_improved_candidate=True
recommend_return_to_finite_q_A4_anisotropy_diagnostic=False
recommend_continue_formula_repair=True

## 诊断状态
- vertex_matches_local_convention;possible_denominator_instability;small_q_continuity_candidate
- vertex_matches_local_convention;possible_denominator_instability;small_q_continuity_not_repaired
- vertex_matches_local_convention;possible_overlap_band_order_or_phase_issue;small_q_continuity_candidate
- vertex_matches_local_convention;small_q_continuity_candidate
- vertex_matches_local_convention;small_q_continuity_not_repaired

## 限制
- gauge_status=prototype_not_ward_verified
- Ward identity / diamagnetic closure 未完成
- n=0 model 未完成
- final_casimir_input=False
- not_final_Casimir_conclusion=True
