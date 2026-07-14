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
python -m validation matsubara positive-orbit-gauss-crosscheck --help
python -m validation matsubara positive-orbit-gauss-scan --help
python -m validation ward commensurate --help
```

## Fixed/composite Gauss 并行约定

`positive-orbit-gauss-crosscheck` 与兼容的 d-wave 入口支持对独立 transverse nodes 做确定性 POSIX-fork 进程并行：

```text
--transverse-workers N
--transverse-task-size M
```

每个子进程继承同一只读 model/evaluator 配置，并计算一个完整 commensurate q orbit。进程内不执行 phase-Hessian pullback、Schur、sheet、reflection 或 logdet。所有节点结果返回父进程后，仍严格按原 Gauss 节点顺序进行 complex Kahan summation。因此 worker 数和 task size 只改变执行调度，不改变积分节点、权重或归并顺序。

进程模式要求操作系统提供 POSIX `fork`。不支持 `fork` 的平台必须使用 `--transverse-workers 1`。启用多个 workers 时，必须把每个子进程中的底层数值库线程数固定为一：

```bash
env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python -m validation matsubara positive-orbit-gauss-crosscheck \
    --transverse-workers 8 \
    --transverse-task-size 4 \
    ...
```

不要仅凭小尺寸等价测试推断性能。正式高阶计算前，应在真实 `nk` 上用较小 Gauss order 做串行/进程 A/B，并同时检查墙钟时间和 shell `time` 的总 CPU 时间。不要同时对多个 q case 再做外层并行。

外部 reference CSV 是可选的。省略 `--reference-csv` 时，命令仍会完成相邻阶数、周期切口、Ward、sheet、reflection 和 passive-logdet 检查。

## 单一方法的逐点分级参数

`positive-orbit-gauss-scan` 同时覆盖 `spm` 与 `dwave` 的正 Matsubara 扇区。所有点始终使用同一个 16-panel composite Gauss-Legendre 方法；不同点只允许改变总 transverse order 和相应预算。

默认阶数按照独立 low/high stage pairs 解释：

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
```

每一级只比较同级高低阶。严格通过的点立即停止；reflection/logdet 已收敛且 sigma 仅轻微超过严格阈值的点，需要满足 soft 门禁和趋势要求，并进行 shifted-periodic-cut 审计。只有困难点才进入后续阶数对。不同 Matsubara 正频共享同一个 microscopic eigensystem batch，因此按 q 选择阶数，而不为了高频点拆分主要计算。

真正的 Matsubara `n=0` 不属于该命令。静态项必须继续使用 exact-static density/stiffness formulation，不能由 `sigma=-K/xi` 外推。

扫描只可能给出 `outer_integral_candidate=True`。它不建立 production response reference，也不代表最终 Casimir energy/torque 已经收敛。

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
