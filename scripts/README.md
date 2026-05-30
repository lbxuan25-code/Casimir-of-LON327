# scripts 入口说明

本目录采用“主题实现目录 + 根目录兼容入口”的结构：

- `normal_state/`：normal-state inspection 与 conductivity 计算。
- `pairing/`：pairing / gap structure 诊断。
- `bdg/`：BdG kernel 与 superconducting response 诊断。
- `response/`：local sheet response、static policy、finite-q response 当前主线。
- `casimir/`：local-response Casimir benchmark 与 convergence runner。
- `archive/`：已完成阶段的旧诊断脚本，仅用于追溯。

根目录下的 `*.py` 文件是兼容 wrapper，用来保持旧命令和旧测试路径可用；新实现优先放在
上面的主题目录里。

## 当前 active 主线

- finite-q response diagnostic：`response/diagnose_finite_q_raw_q0_consistency.py`
- local-response distance scan：`casimir/benchmark_casimir_local_response_distance_scan.py`
- local-response benchmark helpers：`casimir/benchmark_casimir_local_response_integral.py`,
  `casimir/converge_casimir_local_response_integral.py`,
  `casimir/refine_casimir_local_convergence_blockers.py`,
  `casimir/run_casimir_local_convergence_final.py`

这些脚本仍是 diagnostic / benchmark-only，不输出正式 Casimir torque 结论。

## 已归档脚本

- `scripts/archive/numerical_stability/`：normal sampling、response convergence、n=0 sensitivity、
  smoke 等历史数值稳定性脚本。
- `scripts/archive/finite_q_diagnostics/`：旧 finite-q anisotropy、local-limit、
  formula consistency、subspace repair 诊断脚本。
- `scripts/archive/legacy_smoke/`：旧 smoke 入口。

移动清单见 `outputs/archive/ARCHIVE_INDEX.md`。需要沿用旧命令时可继续调用根目录 wrapper。

## 原则

- archive 脚本不要作为当前入口。
- 不删除历史脚本。
- 不修改物理公式。
- 新脚本实现优先进入主题目录；根目录只新增兼容 wrapper。
