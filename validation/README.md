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
- `static`：独立 exact zero-Matsubara k-grid、quadrature 和 d-wave reference；
- `matsubara`：零频与正频总验证、完整 orbit、fixed/composite Gauss、性能 preflight 和分级扫描。

示例：

```bash
python -m validation static nk-scan --nks 8 12 16
python -m validation matsubara orbit-gauss-preflight --help
python -m validation matsubara matsubara-orbit-gauss-crosscheck --help
python -m validation matsubara total-orbit-gauss-scan --help
python -m validation ward commensurate --help
```

## 总 Matsubara 共用 microscopic batch

`matsubara-orbit-gauss-crosscheck`、`orbit-gauss-preflight` 与 `total-orbit-gauss-scan` 都要求真正的 Matsubara `n=0` 与至少一个正频率同时出现。它们在同一次 complete-orbit callback 中共享 midpoint/shifted eigensystems：

- `n=0` 使用 exact static divided-difference factor，随后进入 density/stiffness、static sheet、static reflection 与带 prime 权重的 Lifshitz 项；
- `n>0` 使用正 Matsubara factor，随后进入 conductivity、positive-frequency reflection 与 Lifshitz 项；
- 禁止对 `n=0` 使用 `sigma=-K/xi`；
- 频率数增加不会增加 transverse callback 数，只增加共享本征系统后的轻量 contraction。

扫描输出统一使用 `primary_response`：静态行为 `diag(chi_bar,Dbar_T)`，正频行为 `sigma_tilde`。reflection 与 logdet 始终由对应频率扇区的正确 electrodynamics contract 构造。

## Fixed/composite Gauss 并行约定

总 Matsubara fixed/composite Gauss 支持对独立 transverse nodes 做确定性 POSIX-fork 进程并行：

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
  BLIS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  python -m validation matsubara orbit-gauss-preflight \
    --transverse-workers 8 \
    --transverse-task-size 4 \
    ...
```

不要仅凭小尺寸等价测试推断性能。正式扫描前必须在真实 `nk` 上运行 preflight。它直接执行正式总后端，并检查：

- `spm` 与 `dwave` 的 serial/fork primitive integral 混合绝对–相对等价；
- batched material/q workspace 标识；
- callback 数等于 transverse node 数且不随频率数增加；
- full transverse period、无 symmetry reduction、fork execution strategy；
- 实际 wall-time speedup 与 worker CPU/wall ratio；
- d-wave `n=0` response components 对独立旧 exact-static primitive 路径的等价。

preflight 是代码正确性、执行路径与速度的前置门，不代替正式阶数收敛扫描。可用 `--no-require-physical` 避免让较低 preflight order 的局部收敛性阻断后端验证；Ward、static strict gate、reflection、logdet 和阶数收敛仍由正式总扫描逐点硬性判定。

preflight 写出带当前 Git head 和运行参数的 manifest。正式 scanner 默认拒绝缺失、失败、参数不匹配或 Git head 已变化的 manifest。

## 单一方法的逐点分级参数

`total-orbit-gauss-scan` 同时覆盖 `spm`、`dwave`、真正的 `n=0` 和所有指定正频率。所有点始终使用同一个 full-period 16-panel composite Gauss-Legendre 方法；不同点只允许改变总 transverse order 和相应预算。

默认阶数按照独立 low/high stage pairs 解释：

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
```

每一级只比较同级高低阶。严格通过的 q case 立即停止。正频 reflection/logdet 已收敛且 sigma 仅轻微超过严格阈值时，可按 soft 门禁和趋势要求接受，并执行 shifted-periodic-cut 审计。真正的 `n=0` 不允许 soft 放宽：static response、strict static Ward、reflection 与 logdet 必须达到严格门禁。只有困难 q case 才进入后续阶数对。

阶数按 q case 而不是单个 Matsubara index 选择，因为同一 q 的全部频率共享主要 eigensystem 成本。扫描只可能给出 `outer_integral_candidate=True`；它不建立 production response reference，也不代表最终 Casimir energy/torque 已经收敛。

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

## 当前验证

GitHub Actions run `29304747922` 在 head `f4be83e361ea4716df965adef07e36fca6fc005a` 上完整通过，包括 targeted contracts、全仓测试、总零频/正频 crosscheck、真实 preflight→manifest→scanner 子进程链和旧入口 smoke。

当前 validation 只证明相应数值门禁；在总 Matsubara reference、完整外层积分与最终 energy/torque 报告完成前，不宣称 finite-q Casimir production-ready。
