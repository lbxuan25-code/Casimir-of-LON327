# scripts 入口说明

本目录脚本按当前用途分为 active 入口和 archive 入口。旧诊断脚本已经移动到
`scripts/archive/`，仍可追溯，但不要作为当前默认入口。

## local-response baseline 相关 active 脚本

- `benchmark_casimir_local_response_integral.py`
- `converge_casimir_local_response_integral.py`
- `refine_casimir_local_convergence_blockers.py`
- `run_casimir_local_convergence_final.py`
- `benchmark_casimir_local_response_distance_scan.py`

这些脚本用于 normal sampling、local-response integral convergence、refined convergence
和 distance scan benchmark。它们仍是 benchmark-only，不输出正式 Casimir 结论。

## finite-q response diagnostic 当前主线

- `diagnose_finite_q_raw_q0_consistency.py`

该脚本只做 response 层 diagnostic prototype，不接入 Lifshitz / Casimir，不做 torque 结论。

## 已归档脚本

- `scripts/archive/numerical_stability/`：normal sampling、response convergence、n=0 sensitivity、
  smoke 等历史数值稳定性脚本。
- `scripts/archive/finite_q_diagnostics/`：旧 finite-q anisotropy、local-limit、
  formula consistency、subspace repair 诊断脚本。
- `scripts/archive/legacy_smoke/`：旧 smoke 入口。

移动清单见 `outputs/archive/ARCHIVE_INDEX.md`。测试中仍需要调用旧脚本的路径已经更新到
archive 位置。

## 原则

- archive 脚本不要作为当前入口。
- 不删除历史脚本。
- 不修改物理公式。
- 如未来继续移动脚本，必须同步更新测试路径和归档清单。
