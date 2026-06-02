# finite-q response 当前状态

## 目标

finite-q response 主线的目标是：不改 H0、不改 pairing，只在 response 层检查有限
`q_parallel` 是否能产生可区分 spm/dwave 的角向差异，尤其是 dwave 节点相关的
角向 response 信号。

该阶段不接入 Lifshitz / Casimir 积分，不做 torque 结论。

## 已完成诊断

- `validation/outputs/archive/response/finite_q_anisotropy/`：A4 angular anisotropy 初步诊断。
- `validation/outputs/archive/response/finite_q_local_limit/`：small-q local-limit decomposition。
- `validation/outputs/archive/response/finite_q_formula_consistency/`：vertex、BZ wrapping、denominator、
  overlap 的初步一致性排查。
- `validation/outputs/archive/response/finite_q_subspace_repair/`：near-degenerate subspace projector 与
  raw/stable denominator 对照诊断。
- `validation/outputs/response/finite_q_raw_q0_consistency/`：raw q=0 bubble 与 local components 的
  定义层级一致性诊断。该目录是当前 active finite-q 输出。

## 当前结论

- vertex mismatch 基本排除。
- BZ wrapping 问题基本排除。
- stable denominator mode 没有明显改善 small-q continuity。
- projector overlap 明显小于 eigenstate overlap offdiag，说明单态 overlap 问题多半是
  gauge / band-order rotation，而不是真实子空间混合。
- small-q continuity 仍未整体修复。
- raw q=0 诊断显示 normal 可对齐 local_sigma，但 BdG 的 spm/dwave raw q=0 bubble
  没有对齐 local_sigma、K_para 或 K_total/omega，提示当前主要问题在 BdG formula layer /
  response 层级对齐。
- 旧 finite-q 诊断结果已经归档到 `validation/outputs/archive/response/`，对应旧脚本归档到
  `validation/scripts/finite_q_diagnostics/`。

## 当前下一步

下一步应继续围绕 raw q=0 finite-q bubble 与 BdG local components 的定义层级一致性，
检查 Nambu / prefactor / K_para / K_total / Sigma_SC 的对应关系。当前不建议回到
finite-q A4 anisotropy 物理解读，也不建议进入 finite-q Casimir plumbing。

## 限制

- `gauge_status=prototype_not_ward_verified`
- `final_casimir_input=False`
- `not_final_Casimir_conclusion=True`
- finite-q diamagnetic / Ward closure 未完成
- n=0 model 未完成
