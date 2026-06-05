# BdG 静态规范约定扫描

这是 convention scan 诊断，不是正式 response 公式修改。本扫描不会改变当前
BdG response 实现。它只用试探 prefactor 重新组合已经计算好的 K_para 与
K_dia，用于比较历史和候选 static stiffness convention。当前正式 BdG total
kernel 使用已经由 Peierls/free-energy 验证的 K_dia - K_para convention。

它不是最终 response 公式，不是最终 optical conductivity，不是最终 Casimir
输入，也不包含 finite momentum response。

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_static_gauge_closure.py --quick --scan-kernel-conventions`
benchmark_only=True
local_response=True
convention_scan_diagnostic=True
not_final_response_formula=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

Delta0=0 lowest_omega=0

## 各 convention 的 Stiffness Norm Ratio
- dwave, current: candidate_stiffness_norm_ratio=1.30144
- spm, current: candidate_stiffness_norm_ratio=1.30144
- dwave, minus_para: candidate_stiffness_norm_ratio=0.698565
- spm, minus_para: candidate_stiffness_norm_ratio=0.698565
- dwave, half_para: candidate_stiffness_norm_ratio=1.15072
- spm, half_para: candidate_stiffness_norm_ratio=1.15072
- dwave, minus_half_para: candidate_stiffness_norm_ratio=0.849282
- spm, minus_half_para: candidate_stiffness_norm_ratio=0.849282
- dwave, minus_dia: candidate_stiffness_norm_ratio=0.698565
- spm, minus_dia: candidate_stiffness_norm_ratio=0.698565
- dwave, minus_para_minus_dia: candidate_stiffness_norm_ratio=1.30144
- spm, minus_para_minus_dia: candidate_stiffness_norm_ratio=1.30144
- dwave, half_both: candidate_stiffness_norm_ratio=0.650718
- spm, half_both: candidate_stiffness_norm_ratio=0.650718
- dwave, minus_half_para_half_dia: candidate_stiffness_norm_ratio=0.349282
- spm, minus_half_para_half_dia: candidate_stiffness_norm_ratio=0.349282

## 最优候选
best_kind=spm
best_convention=minus_half_para_half_dia
best_para_prefactor=-0.5
best_dia_prefactor=0.5
best_candidate_stiffness_norm_ratio=0.349282
legacy_threshold=0.001
legacy_passes_threshold=False

本扫描不声称 static gauge closure 已解决，也不选择最终 optical conductivity。
正式 response contract 是单独验证过的 K_dia - K_para convention。
