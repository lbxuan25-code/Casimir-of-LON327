# finite-q q=0 BdG kernel-stack consistency 诊断摘要

本轮目标是检查 q=0 finite-q response 层级是否与已有 local BdG kernel 定义一致。
优先级为 K_para(q=0) 对 local K_para，再到 K_total(q=0) 对 local K_total，最后是 Sigma_SC(q=0)=K_total/omega。
raw_q0_bubble 仅作为兼容字段保留，含义等同 K_para_q0，不再解释为 Sigma_SC 或 conductivity。

kinds=['normal', 'spm', 'dwave']
matsubara_list=[1]
nk_list=[6, 8]
denominator_mode_list=['raw', 'stable']
deg_tol_list=[1e-08, 1e-07]
temperature=30.0
delta0=0.04
eta=0.0001

normal_q0_consistency_pass=True
normal_min_error_raw_to_local_sigma=2.77234e-05
normal_min_error_hook_to_local_sigma=0
bdg_K_para_q0_consistent=True
bdg_K_total_q0_consistent=True
bdg_Sigma_SC_q0_consistent=True
bdg_min_error_K_para_q0_to_local_K_para=0
bdg_min_error_K_total_q0_to_local_K_total=0
bdg_min_error_Sigma_SC_q0_to_local_K_total_over_omega=0
raw_q0_unmatched=True
spm_best_match=local_K_para (0)
dwave_best_match=local_K_para (0)
formula_layer_mismatch_detected=True
recommend_normal_kubo_formula_repair=False
recommend_bdg_kernel_stack_alignment=False
recommend_formula_rederive=True
recommend_return_to_small_q_smoothness_diagnostic=True
finite_q_dia_statuses=['not_applicable', 'q0_fallback_only']
ward_statuses=['not_closed']
any_valid_for_casimir_input=False

## formula_layer_diagnosis
- bdg_K_para_q0_consistent
- normal_finite_q_formula_mismatch
- normal_sigma_like

## 限制
- 当前 finite-q response 仍不是 Ward 完备
- finite-q diamagnetic 目前只提供 q0_fallback_only
- Ward closure 未完成，ward_status=not_closed
- n=0 model 未完成
- valid_for_casimir_input=False
- final_casimir_input=False
- not_final_Casimir_conclusion=True
