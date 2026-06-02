# finite-q raw q=0 formula consistency 诊断摘要

本轮目标是检查 raw q=0 finite-q bubble 与已有 local response 的定义层级是否一致。
q=0 hook 会直接返回 local reference；raw q=0 bubble 则强制走与 q>0 相同的 finite-q bubble 公式。

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
bdg_raw_q0_sigma_like=False
bdg_raw_q0_para_like=False
bdg_raw_q0_total_over_omega_like=False
raw_q0_unmatched=True
spm_best_match=local_sigma (0.629737)
dwave_best_match=local_sigma (0.62431)
formula_layer_mismatch_detected=True
recommend_normal_kubo_formula_repair=False
recommend_bdg_layer_alignment=True
recommend_formula_rederive=True
recommend_return_to_small_q_smoothness_diagnostic=False

## formula_layer_diagnosis
- normal_finite_q_formula_mismatch
- normal_sigma_like
- raw_q0_unmatched

## 限制
- 当前 finite-q response 仍不是 Ward 完备
- finite-q diamagnetic / Ward closure 未完成
- n=0 model 未完成
- final_casimir_input=False
- not_final_Casimir_conclusion=True
