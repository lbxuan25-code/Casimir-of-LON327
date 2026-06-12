# Stage 4.20 User-run targeted Ward refinement scan

## Purpose

Stage 4.20 provides a user-run targeted refinement scan for the Stage 4.19 worst-case Ward residual cluster. It focuses on low temperature, Matsubara index 1, diagonal q, refinement level, Gauss order, and Fermi-window sensitivity.

## Why User-Run

The targeted and confirm presets can be expensive, especially at adaptive level 5 and Gauss order 5. Codex should only run lightweight tests; full targeted scans should be launched by the user in a local terminal.

## Boundary

- no main response change
- no bubble sign change
- no direct contact change
- no source/observable change
- no residual tuning
- no fitted contact
- no E_ET added
- no conductivity / reflection / Casimir
- no Casimir-ready claim

## Presets

- `quick`: lightweight smoke-test preset.
- `worst-only`: only the Stage 4.19 worst q cluster around `q_diag_pos`, `q_scale=1.0`.
- `targeted`: focused scan over `q_diag_pos/q_diag_neg`, `q_scale=1.0/0.5`, levels `3,4,5`, orders `3,5`, and Fermi windows `0.03,0.05,0.08,0.12`.
- `confirm`: higher-refinement confirmation subset using levels `4,5`.
- `custom`: starts from quick and applies CLI overrides.

## Recommended Terminal Commands

```bash
# 建议先设置，避免每个进程内部 BLAS 再开多线程
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 查看将要跑多少 case，不真正计算
python validation/scripts/response/stage4_20_user_run_targeted_refinement_scan.py --preset targeted --workers 8 --dry-run

# 先跑最差 case 小集合
python validation/scripts/response/stage4_20_user_run_targeted_refinement_scan.py --preset worst-only --workers 8 --resume

# 如果时间可接受，再跑 targeted
python validation/scripts/response/stage4_20_user_run_targeted_refinement_scan.py --preset targeted --workers 8 --resume

# 若 targeted 太重，只跑前 24 个 case
python validation/scripts/response/stage4_20_user_run_targeted_refinement_scan.py --preset targeted --workers 8 --max-cases 24 --resume
```

## Parallel Execution Notes

The script uses `concurrent.futures.ProcessPoolExecutor`. `--workers` defaults to `max(1, os.cpu_count() - 1)`. Each case is independent and returns one result row.

The script sets BLAS-related thread environment variables with `setdefault`, so existing shell values take precedence.

## Resume / Checkpoint Notes

With `--resume`, the script reads completed case keys from the output JSON and checkpoint JSONL. Completed cases are skipped. A checkpoint row is appended after each finished case.

The case key includes temperature, Matsubara index, q case, q scale, adaptive level, Gauss order, Fermi window, and coarse grid.

## Expected Runtime Control

Use `--dry-run` first to inspect the planned case count and rough quadrature-point upper bounds. Use `--max-cases N` for partial runs.

## Output Interpretation

Case status:

- `CLOSED`: `max_corrected_norm < 1e-6`
- `ACCEPTABLE_BUT_MONITOR`: `1e-6 <= max_corrected_norm < 1e-5`
- `NOT_CLOSED`: `max_corrected_norm >= 1e-5`

Global status:

- `TARGETED_REFINEMENT_PASSED`
- `TARGETED_REFINEMENT_MOSTLY_PASSED`
- `NEEDS_HIGHER_REFINEMENT_OR_WINDOW`
- `POSSIBLE_NON_QUADRATURE_REMAINING_ISSUE`

## Next Step

If targeted refinement passes, the next stage may proceed to an independent response-to-conductivity validation. Passing this script is still not a conductivity, reflection, or Casimir conclusion.

