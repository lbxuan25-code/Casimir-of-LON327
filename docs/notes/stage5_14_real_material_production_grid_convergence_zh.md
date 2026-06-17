# Stage 5.14 real-material production-grid energy convergence

## 目标

Stage 5.14 读取 Stage 5.13 的 zero-mode 与 grid-convergence planning audit，并执行真实材料 production-grid energy convergence 的三档网格检查：

```text
coarse: n_max=8,  n_Q=16, n_phi=8
medium: n_max=16, n_Q=24, n_phi=12
fine:   n_max=32, n_Q=32, n_phi=16
```

输入必须具有：

```text
STAGE5_13_ZERO_MODE_GRID_CONVERGENCE_AUDIT_PASSED
```

## 边界

本阶段不修改 response formula、bubble sign、direct contact、Ward convention、\(\Pi\to\sigma\)、unit conversion、reflection convention 或 trace-log convention。

输出只用于 production-grid energy convergence gate，不输出 force，不输出 torque，也不声称 final physical result。

## \(Q=0\) 与 \(n=0\)

\(Q=0\) 不作为普通 grid point。径向 \(Q\) 网格使用 interior quadrature nodes，因此不会把 \(Q=0\) endpoint 放进普通 TE/TM 方向网格。

\(n=0\) 使用 \(\xi\to0^+\) extrapolated \(R^{TE/TM}\) 代理，不直接对 \(\Omega=0\) 使用 \(\sigma=-\Pi/\Omega\)。实现上，\(n=0\) 能量项使用相同 \((Q,\phi)\) 的第一个正 Matsubara reflection row 作为外推代理，并保留 Matsubara prime weight \(1/2\)。

## cache 与断点续算

Stage 5.14 复用 Stage 5.11 的 material response/reflection point cache。CLI 支持：

```text
--workers
--cache-dir
--resume
--skip-existing
--force-recompute
```

推荐正式长任务运行前限制 BLAS/OpenMP 线程数，然后用 `--workers` 控制点级多进程：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python validation/scripts/response/stage5_14_real_material_production_grid_convergence.py \
  --workers 8 \
  --resume \
  --skip-existing \
  --cache-dir validation/outputs/response/material_reflection_grid/cache
```

## 收敛判据

脚本记录 coarse→medium 和 medium→fine 的相对变化：

```text
relative_change = |E_b - E_a| / max(|E_a|, |E_b|)
```

判据：

```text
medium→fine < 5%     PASS
5% <= medium→fine < 15%  MONITOR
medium→fine >= 15%   FAIL
```

若出现缺点、非有限数值或其他数值异常，则 energy convergence 判为 FAIL。

## 输出

默认输出：

```text
validation/outputs/response/material_grid_convergence/stage5_14_real_material_production_grid_convergence.json
validation/outputs/response/material_grid_convergence/stage5_14_real_material_production_grid_convergence.md
```

JSON 包含 boundary、input、grid_runs、energy_convergence、cache_summary、checks 和 diagnostic_status。
