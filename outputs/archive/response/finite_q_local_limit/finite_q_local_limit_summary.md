# finite-q local-limit 分解诊断摘要

本轮目的：拆解 finite-q bubble 在 q->0 时与 local response 不连续的来源，判断它更接近
local sigma、BdG K_para、K_total、K_total/omega，还是 normal Kubo sigma。

本轮不做 Casimir、不做 torque、不做正式物理结论。q=0 local hook 只是直接引用
local reference；small-q finite-q bubble continuity 才是连续极限诊断。

kinds=['normal', 'spm', 'dwave']
matsubara_list=[1]
small_q_list=[0.0001, 0.001]
q_phi_list=[0.0, 0.7853981634]
nk_list=[6]
temperature=30.0
delta0=0.04
eta=0.0001

global_best_match_component=local_sigma
best_match_components_at_min_q_max_nk=['local_sigma']
best_match_relative_error_at_min_q_max_nk=2.77213e-05
local_limit_component_match_candidate=True
small_q_error_monotonic_in_q=False
error_improves_with_nk=False
likely_missing_contact_or_diamagnetic_completion=False
likely_formula_or_vertex_mismatch=True
worth_next_finite_q_casimir_plumbing_smoke=False

## 诊断状态
- best_matches_local_sigma;warning_small_q_not_smooth;likely_formula_or_vertex_mismatch

## 建议
recommend_finite_q_formula_repair=True

## 限制
- gauge_status=prototype_not_ward_verified
- Ward identity / diamagnetic closure 未完成
- n=0 model 未完成
- final_casimir_input=False
- not_final_Casimir_conclusion=True
