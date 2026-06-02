# finite-q 当前阶段索引

当前 finite-q response 仍是 diagnostic prototype，不是 Ward 完备 response，也不是最终
Casimir input。

## 目录说明

- `validation/outputs/archive/response/finite_q_anisotropy/`：A4 初步诊断。已看到 finite-q angular anisotropy 和
  A4_pairing_contrast 的 quick 信号，但 small-q continuity 未通过，因此不能做物理解释。
- `validation/outputs/archive/response/finite_q_local_limit/`：local-limit decomposition。用于判断 finite-q bubble 在
  `q->0` 时最接近哪个 local component。
- `validation/outputs/archive/response/finite_q_formula_consistency/`：vertex / BZ wrapping / denominator / overlap 初排查。
  当前 vertex mismatch 和 BZ wrapping 基本排除，但 small-q continuity 未整体修复。
- `validation/outputs/archive/response/finite_q_subspace_repair/`：projector / denominator repair 诊断。projector overlap
  表明单态 overlap 问题多半是 gauge / band-order rotation；stable denominator 没有明显改善。
- `validation/outputs/response/finite_q_raw_q0_consistency/`：当前关键 active 诊断入口。用于比较 raw q=0 bubble 与
  local_sigma、K_para、K_total/omega 的定义层级。

## 当前判断

当前不建议进入 finite-q Casimir plumbing，也不建议解释 A4 signal 为真实 torque 来源。
下一步应继续检查 BdG formula layer / response 层级对齐。
