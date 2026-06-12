# Stage 4.20 user-run targeted refinement scan

## 目的

Stage 4.20 提供一个用户本地运行的 targeted Ward refinement scan，用于继续检查 Stage 4.19 中低温、低 Matsubara、斜向 \(\mathbf q\)、低 refinement 或窄 Fermi window 的 worst-case cluster。

本阶段不扩大默认全参数扫描，也不由 Codex 运行重任务。脚本支持多核并行、断点续跑、dry-run 和轻量测试。

## 边界

- 不修改主 physical response formula。
- 不修改 bubble prefactor sign。
- 不修改 \(V_i\)、\(M_{ij}\)、\(j_i=-V_i\)。
- 不修改 source/observable split。
- 不修改 direct contact。
- 不新增 fitted contact。
- 不新增 \(E^{ET}\)。
- 不进入 conductivity、reflection 或 Casimir。
- 不声明 Casimir-ready。

## residual convention

脚本使用 Stage 4.18 固化后的 corrected residual：

\[
R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu},
\]

\[
R_R[\mu]=i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.
\]

## 运行方式

默认 `--preset quick` 只用于轻量测试。重任务由用户在终端显式运行：

```bash
python validation/scripts/response/stage4_20_user_run_targeted_refinement_scan.py --preset worst-only --workers 8 --resume
python validation/scripts/response/stage4_20_user_run_targeted_refinement_scan.py --preset targeted --workers 8 --resume
```

建议先用 `--dry-run` 查看 case 数和积分点上界，再决定是否运行。

## BLAS 线程

脚本使用 `os.environ.setdefault` 设置 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 和 `NUMEXPR_NUM_THREADS` 为 `1`，避免每个 Python 进程内部再开多线程。若用户已经在 shell 中设置这些变量，shell 设置优先。

