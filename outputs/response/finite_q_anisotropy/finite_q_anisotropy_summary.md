# finite-q response 角向各向异性诊断摘要

本阶段选择 finite-q response，是为了检查 Casimir 几何中的有限 q_parallel 是否能在
不修改 H0、不修改 pairing 的情况下放大 spm/dwave 的 response 层差异，尤其是
dwave 节点相关的角向响应差异。

本脚本只做 response 层 finite-q diagnostic / prototype，不做 Casimir torque，
也不接入正式 Lifshitz 积分。

q_magnitude 当前使用 dimensionless BZ momentum，与 k 网格单位一致，不是 SI wavevector。

kinds=['normal', 'spm', 'dwave']
matsubara_list=[1]
q_list=[0.0, 0.05]
q_phi_list=[0.0, 0.7853981634]
temperature=30.0
nk=6
delta0=0.04
eta=0.0001

q_to_0_local_limit_passed=True
finite_q_angular_anisotropy_signal=True
dwave_normal_vs_spm_normal_contrast_signal=True
worth_next_finite_q_casimir_plumbing_smoke=True

## 每个 q 的诊断
- q=0
  normal_max_abs_A4_xx=0
  spm_max_abs_A4_xx=0
  dwave_max_abs_A4_xx=0
  max_abs_contrast_dwave_minus_spm=0.00346217
- q=0.05
  normal_max_abs_A4_xx=8.96465e-05
  spm_max_abs_A4_xx=0.000220251
  dwave_max_abs_A4_xx=4.7538e-05
  max_abs_contrast_dwave_minus_spm=0.0128194

## 限制
- gauge_status=prototype_not_ward_verified
- finite-q diamagnetic/Ward identity 尚未严格闭合
- n=0 zero-frequency model 仍未完成
- final_casimir_input=False
- not_final_Casimir_conclusion=True
