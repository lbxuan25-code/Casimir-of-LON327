# Validation

`validation/` 保存能够直接消费当前 two-band finite-q contract 的数值验证入口、可测试实现和轻量证据。根目录不再放置独立 runner 或 analyzer。

## 统一入口

所有 active validation 命令通过：

```bash
python -m validation --help
python -m validation <group> <command> [options]
```

当前分组：

- `ward`：finite-q Ward、collective phase-column 与 phase-Hessian 诊断；
- `static`：exact zero-Matsubara k-grid、quadrature 和 d-wave reference 扫描；
- `matsubara`：正 Matsubara 单点、完整 orbit、panel 与 fixed/composite Gauss 验证。

示例：

```bash
python -m validation static nk-scan --nks 8 12 16
python -m validation matsubara positive-point --help
python -m validation matsubara dwave-orbit-panel-adaptive --help
python -m validation matsubara dwave-orbit-gauss-crosscheck --help
python -m validation ward commensurate --help
```

## Fixed/composite Gauss 并行约定

`dwave-orbit-gauss-crosscheck` 支持对独立 transverse nodes 做确定性线程并行：

```text
--transverse-workers N
--transverse-task-size M
```

每个 worker 仍计算一个或一组完整 commensurate q orbit。worker 内不执行 bond metric、Schur、sheet、reflection 或 logdet。所有节点结果返回父线程后，严格按原 Gauss 节点顺序进行 complex Kahan summation。因此 worker 数和 task size 只改变执行调度，不改变积分节点、权重或归并顺序。

启用多个 transverse workers 时，必须把底层数值库线程数固定为一，避免线程过度订阅：

```bash
env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python -m validation matsubara dwave-orbit-gauss-crosscheck \
    --transverse-workers 8 \
    --transverse-task-size 4 \
    ...
```

推荐先使用 `workers=8, task_size=4`。内存不足、温度过高或持续降频时降低 workers；任务数较少时 integrator 会自动把有效 worker 数限制为 task 数。不要同时对多个 q case 再做外层并行，否则容易发生 CPU 和内存带宽过度订阅。

外部 reference CSV 是可选的。省略 `--reference-csv` 时，命令仍会完成：

- 同一切口下相邻总阶数比较；
- 同一阶数下周期切口一致性比较；
- Ward、sheet、reflection 和 passive-logdet 物理管线验证。

只有需要与一个独立已保存结果比较时才传入 `--reference-csv`。

## 目录边界

- `commands/`：CLI 参数、文件读写和终端输出；不承载新的物理公式；
- `lib/`：可测试的数值算法、诊断分解和 validation model adapters；
- `outputs/`：每个验证对象的复现命令、README 和轻量 summary；
- `reports/`：跨验证对象的当前状态与 artifact policy；
- `__main__.py`：稳定的公共命令路由。

新增诊断时，应优先扩展现有 group 和 backend，而不是新增版本化平行实现，或在 `validation/` 根目录新增 `run_*.py`、`analyze_*.py`、`average_*.py`。

## 输出约定

- 完整 CSV、JSON、txt、log、figure 和中间数组均视为可复现本地产物，默认由 Git 忽略；
- Git 中只保留 README、command、summary、status 和必要的小型 machine-readable convergence 摘要；
- 被新诊断取代的历史实现和输出直接删除，由 Git history 追溯，不建立 archive 目录；
- 正式主计算结果只有在 production reference 建立后才进入根目录 `outputs/`；
- 不使用根目录 `results/`。

本地清理时先预览，再只删除被 Git 忽略的 validation 产物：

```bash
git clean -ndX validation/outputs
git clean -fdX validation/outputs
```

正在运行的任务所写目录不得在任务结束前清理。需要保留的 compact summary 应先整理为带 `summary` 或 `status` 的文件，再执行清理。

当前 validation 只证明相应数值门禁；在 exact-static reference、正频连续性和外层积分报告全部完成前，不宣称 finite-q Casimir production-ready。
