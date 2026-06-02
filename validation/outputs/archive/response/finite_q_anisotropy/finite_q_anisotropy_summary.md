# refined finite-q response 角向各向异性诊断摘要

本轮是 refined finite-q response diagnostic，用于检查 Casimir 几何中的有限 q_parallel 是否能在
不修改 H0、不修改 pairing 的情况下放大 spm/dwave 的 response 层差异，尤其是
dwave 节点相关的角向响应差异。

本脚本只做 response 层 finite-q diagnostic / prototype，不做 Casimir torque，
也不接入正式 Lifshitz 积分。
q=0 local reference hook 与真正 small-q finite-q bubble continuity 是两件事；
主 pairing contrast 现在使用 A4_pairing_contrast，不再使用 legacy_response_xx_contrast。

q_magnitude 当前使用 dimensionless BZ momentum，与 k 网格单位一致，不是 SI wavevector。

kinds=['normal', 'spm', 'dwave']
matsubara_list=[1]
q_list=[0.0, 0.05]
small_q_list=[0.0001, 0.001]
q_phi_list=[0.0, 0.7853981634]
temperature=30.0
nk=6
delta0=0.04
eta=0.0001

q0_local_reference_hook_passed=True
small_q_finite_q_bubble_continuity_passed=True
warning_small_q_not_smooth=True
finite_q_angular_anisotropy_signal=True
A4_pairing_contrast_signal=True
worth_next_finite_q_casimir_plumbing_smoke=True

## 每个 q 的诊断
- q=0
  normal_max_abs_A4_xx=0
  spm_max_abs_A4_xx=0
  dwave_max_abs_A4_xx=0
  max_abs_delta_A4_spm=0
  max_abs_delta_A4_dwave=0
  max_abs_A4_pairing_contrast=0
  max_abs_A4_trace_pairing_contrast=0
- q=0.05
  normal_max_abs_A4_xx=8.96465e-05
  spm_max_abs_A4_xx=8.12678e-05
  dwave_max_abs_A4_xx=1.78689e-05
  max_abs_delta_A4_spm=8.37872e-06
  max_abs_delta_A4_dwave=7.17776e-05
  max_abs_A4_pairing_contrast=6.33989e-05
  max_abs_A4_trace_pairing_contrast=2.24441e-07

## small-q continuity
small_q_min_relative_error=1.18542e-09
small_q_max_relative_error=2.77213e-05
- good_continuity_candidate: 6

## 限制
- gauge_status=prototype_not_ward_verified
- finite-q diamagnetic/Ward identity 尚未严格闭合
- n=0 zero-frequency model 仍未完成
- final_casimir_input=False
- not_final_Casimir_conclusion=True
