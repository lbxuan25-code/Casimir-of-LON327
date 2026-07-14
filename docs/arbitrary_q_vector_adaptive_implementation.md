# arbitrary-q vector-adaptive 实施与性能结构说明

## 当前状态

本分支已经加入可运行的 arbitrary-q vector-adaptive cubature 后端，但它仍是 diagnostic candidate。

```text
fixed shifted periodic N×N grid:
  retained formal reference

vector-adaptive hierarchical cubature:
  implemented and tested
  diagnostic candidate only
  not formally qualified
```

不得因为代码已经存在而修改：

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

## 实现文件

```text
src/lno327/workflows/arbitrary_q_vector_adaptive.py
src/lno327/workflows/arbitrary_q_vector_adaptive_cached.py
src/lno327/workflows/arbitrary_q_vector_adaptive_parallel.py
validation/commands/matsubara/arbitrary_q_vector_adaptive_compare.py
```

## 保留的物理合同

adaptive 后端没有复制或修改 finite-q 物理公式。它继续使用当前 arbitrary-q 主链的：

```text
exact q_crystal = R(-theta) q_lab
one established finite-q q workspace
one established packed primitive kernel
exact n=0 divided differences
positive Matsubara batch
RHS-aware Ward source
phase-Hessian pullback after integration
amplitude/phase Schur after integration
sheet / reflection / passive logdet downstream
```

每个 cell 只计算线性 primitive。所有 accepted high-rule cell primitives 先按稳定 cell 顺序使用 complex Kahan 求和，完整 BZ 积分后才执行 phase-Hessian 和 Schur。

## Paired low/high tensor-Gauss rules

每个矩形 cell 使用一对 **non-embedded** tensor-Gauss rules。默认 GL2 与 GL3 的节点不构成包含关系，因此这里不是 nested 或 Gauss-Kronrod rule。

当前实现把同一批 cell 的 low/high 节点合并到一个 material/q workspace：

```text
one cell batch
  -> low points + high points
  -> one material workspace
  -> one q workspace
  -> one shifted-eigensystem batch
  -> one all-Matsubara Kubo-factor batch
  -> variable slices recover each cell's low/high primitives
```

这不会减少实际积分节点数，但会把每个 cell batch 的 q-workspace、shifted-eigh、Kubo-factor 和 operator-diagnostic 调用从两次降到一次。

标准 reference-square Gauss points/weights 按 order 缓存；每个 cell 只执行 affine mapping 和 Jacobian scaling。

## 自适应控制

refinement score 来自：

```text
all electromagnetic primitive blocks
all collective primitive blocks
contact/direct terms
analytic Ward RHS
all requested Matsubara frequencies
primitive Ward residual difference
```

一个 q 的 exact zero 和所有正 Matsubara 共用：

```text
one adaptive cell tree
one combined low/high point batch per cell group
one shifted eigensystem workspace per combined cell group
```

停止条件由：

```text
relative_tolerance
absolute_tolerance
ward_error_tolerance
max_level
max_iterations
refine_fraction
max_cells
max_evaluation_points
```

共同控制。每轮 profile 记录 active/selected cells、最大/中位/p90 cell score、全局误差比、Ward 误差比和新增点数，供后续决定是否修改 refinement fraction。

未达到误差门时默认抛出 `AdaptiveConvergenceError`，不得静默接受预算耗尽结果。

## Response cache 的 fail-closed 合同

nonconverged diagnostic result 只可在调用显式设置 `require_converged=False` 时返回和缓存。

```text
diagnostic nonconverged cache
  + later require_converged=True
  -> AdaptiveConvergenceError

new strict nonconverged evaluation
  -> raise before cache insertion
```

因此 response cache 不能绕过严格收敛门。

## Material-node cache 与 per-call profile

`ReusableHierarchicalMaterialNodeCache` 使用 exact IEEE-754 k-node keys。第一次遇到节点时缓存：

```text
midpoint energies
midpoint eigenstates
midpoint occupations
q=0 eta2-eta2 static-bubble integrand
```

Goldstone/HS counterterm 不再逐 cell 调用旧 `hs_counterterm()` 重复对角化，而由缓存的 q=0 phase-bubble node integrands 做确定性加权求和。

`ArbitraryQVectorAdaptiveProfile-v2` 中 cache 计数全部是本次调用增量：

```text
node hits/misses
midpoint eigensystem builds
material batch builds
counterterm-node hits/misses
q=0 counterterm workspace builds
```

cache 生命周期累计值另存于：

```text
cache_totals_after_call
metadata["node_cache"]
```

profile 还区分：

```text
primitive_integration_seconds
postprocess_seconds
total_seconds
```

## Cache 资源上限

可选参数：

```text
max_cache_nodes
max_cache_bytes
```

采用 fail-closed/no-eviction 策略。达到上限时抛出 `MemoryError`，不会静默丢节点或改变数值定义。正式 outer 前仍需根据目标硬件和 q 批次冻结预算与 epoch/reset policy。

## POSIX-fork 并行

`ArbitraryQVectorAdaptiveParallelEvaluator` 在创建 fork pool 前预热确定性的初始 low/high node union：

```text
parent prewarm midpoint eigensystems
parent prewarm q=0 counterterm integrands
fork
workers inherit read-mostly cache through copy-on-write
```

metadata 记录：

```text
parent_prewarm_seconds
parent_prewarm_nodes
worker_pid
per-task worker cache delta
per-worker final cache snapshot
worker RSS/PSS
actual BLAS threadpool
```

q-level 并行继续使用 persistent POSIX fork、单线程 BLAS/OpenMP、有序 task return 和 compact payload。

## 公平的 fixed/adaptive 诊断计时

```bash
python -m validation diagnostic arbitrary-q-vector-adaptive-compare \
  --pairing dwave \
  --q-model 0.0300152 0.0200101 \
  --matsubara-indices 0 1 \
  --fixed-N 128 \
  --coarse-grid 6 \
  --low-order 2 \
  --high-order 3 \
  --adaptive-rtol 1e-3 \
  --adaptive-atol 1e-9 \
  --adaptive-ward-atol 1e-9
```

命令分别报告：

```text
fixed grid build
fixed material build
fixed warm response
fixed primitive/postprocess estimate

adaptive cache-object build
adaptive cold response including lazy node construction
adaptive warm new-q response using material-node cache
adaptive exact-response-cache hit
adaptive primitive integration
adaptive postprocessing

fixed/adaptive physical pipeline
primitive/response/reflection/logdet agreement
```

exact-response cache hit 只用于验证缓存开销，不得被当作正常 outer 新 q 的成本。

## 固定网格没有被替换

fixed-grid 后端继续承担：

```text
formal arbitrary-q reference
adaptive numerical comparison target
fallback when adaptive fails or reaches budget
outer-integration qualification baseline
```

vector-adaptive 在完成独立 formal policy、performance preflight 和 fixed-grid/complete-orbit numerical qualification前，不得变成默认 quadrature backend。

## 已有测试

测试覆盖：

```text
spm and dwave
adaptive initial high rule vs one-shot shared primitive kernel
combined low/high one-q-workspace count
counterterm vs established ansatz hs_counterterm
no midpoint eigensystem rebuild for cached counterterm
per-call cache delta across q
cell-batch-size deterministic equality
strict nonconvergence does not pollute response cache
cached diagnostic nonconvergence cannot bypass strict request
two-plate exact-q response-cache reuse
reference tensor-Gauss cache and non-embedded relation
cache-node budget fail closed
parent prewarm before fork
worker PID/cache telemetry
serial vs POSIX-fork equality
validation CLI diagnostic-only routing
```

## 后续资格化

要将 adaptive 提升为 production candidate，仍需：

1. 冻结 `ArbitraryQVectorAdaptiveFormalPolicy`，包括误差容差、资源预算、cell batching、cache budget 和工作负载。
2. 在同一 clean head 上运行 real-hardware adaptive performance preflight。
3. 对 spm/dwave 的 axis、generic、near-diagonal、exact-diagonal 和 rotated q 比较 complete-orbit、fixed N=256/384/512 与 adaptive tolerance ladder。
4. 扫描 `cell_batch_size = 64,128,256,512`，同时记录 wall、stage timing、RSS/PSS 与 cache bytes。
5. 对真实 two-plate common-lab logdet 做 tolerance refinement 和独立 adaptive-tree audit。
6. 建立 outer 所需 q-envelope、tail convergence 和长期 cache epoch/reset policy。

完成这些证据前，不得使用 adaptive 结果作为正式完整 Casimir 积分输入。
