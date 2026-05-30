# scripts 入口说明

本目录脚本按当前用途分为 active 入口、finite-q response diagnostic、local-response
baseline、历史稳定性和辅助检查。当前优先建立清晰索引，不移动仍被 tests 调用的脚本。

## local-response baseline 相关 active 脚本

- `benchmark_normal_fs_adaptive_integration.py`
- `benchmark_casimir_local_response_integral.py`
- `converge_casimir_local_response_integral.py`
- `refine_casimir_local_convergence_blockers.py`
- `run_casimir_local_convergence_final.py`
- `benchmark_casimir_local_response_distance_scan.py`

这些脚本用于 normal sampling、local-response integral convergence、refined convergence
和 distance scan benchmark。它们仍是 benchmark-only，不输出正式 Casimir 结论。

## finite-q response diagnostic 当前主线

- `diagnose_finite_q_response_anisotropy.py`
- `diagnose_finite_q_local_limit_decomposition.py`
- `diagnose_finite_q_formula_consistency.py`
- `diagnose_finite_q_subspace_denominator_repair.py`
- `diagnose_finite_q_raw_q0_consistency.py`

这些脚本只做 response 层 diagnostic prototype，不接入 Lifshitz / Casimir，不做 torque 结论。

## 历史稳定性和辅助脚本

早期 response convergence、normal sampling、BdG kernel、n=0 sensitivity、pairing/gap
inspection 等脚本保留用于可追溯诊断。它们不是当前推荐入口，除非需要复查对应历史问题。

## archive 说明

`scripts/archive/` 下脚本不要作为当前入口。当前没有移动脚本；如果未来要移动一次性历史脚本，
必须先确认没有 active tests 或 active scripts import，并在本 README 记录旧路径和新路径。
