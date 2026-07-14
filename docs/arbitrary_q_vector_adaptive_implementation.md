# arbitrary-q vector-adaptive 实施说明

## 当前状态

本分支已经加入一个可运行的 arbitrary-q vector-adaptive cubature 后端，但它目前仍是 diagnostic candidate。

```text
fixed shifted periodic N×N grid:
  retained formal reference

vector-adaptive hierarchical cubature:
  implemented and tested
  diagnostic candidate only
  not formally qualified
```

不得因为代码已经存在而修改以下状态：

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

每个 cell 上只能计算线性 primitive。所有 accepted high-rule cell primitives 先按稳定 cell 顺序使用 complex Kahan 求和，完整 BZ 积分后才执行 phase-Hessian 和 Schur。

## 自适应控制

每个矩形 cell 同时计算 low-order 和 high-order tensor-Gauss rules。refinement score 来自：

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
one point set per rule evaluation
one shifted eigensystem workspace per batched cell group
```

自适应停止条件由以下参数共同控制：

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

未达到误差门时默认抛出 `AdaptiveConvergenceError`，不得静默接受预算耗尽结果。

## 缓存与性能

`ReusableHierarchicalMaterialNodeCache` 使用 exact IEEE-754 k-node keys。第一次遇到节点时缓存：

```text
midpoint energies
midpoint eigenstates
midpoint occupations
q=0 eta2-eta2 static-bubble integrand
```

随后这些数据跨 q、角度和 Matsubara batch 复用。

Goldstone/HS counterterm 不再逐 cell 调用旧 `hs_counterterm()` 重复对角化。counterterm 由缓存的 q=0 phase-bubble node integrands 做确定性加权求和；测试要求它与 established ansatz counterterm 数值一致，并要求：

```text
counterterm shifted-eigh calls = 0
second q midpoint-eigh increment = 0
second q q=0-counterterm-workspace increment = 0
```

q-level 并行使用：

```text
ArbitraryQVectorAdaptiveParallelEvaluator
persistent POSIX fork pool
single-thread BLAS/OpenMP
ordered task return
compact payload
```

## 固定网格没有被替换

当前 fixed-grid 后端继续承担：

```text
formal arbitrary-q reference
adaptive numerical comparison target
fallback when adaptive fails or reaches budget
outer-integration qualification baseline
```

vector-adaptive 在完成独立 formal policy、performance preflight 和 fixed-grid/complete-orbit numerical qualification 前，不得变成默认 quadrature backend。

## 诊断命令

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

命令比较：

```text
packed primitives
post-Schur primary response
integrated Ward / strict static physical gates
reflection
logdet
fixed point count
adaptive point evaluations
accepted cells
stage timings
material/cache counters
```

输出明确保持：

```text
adaptive_backend_promoted = False
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

## 已有测试

测试覆盖：

```text
spm and dwave
adaptive initial high rule vs one-shot shared primitive kernel
counterterm vs established ansatz hs_counterterm
no midpoint eigensystem rebuild for cached counterterm
hierarchical node reuse across q
cell-batch-size deterministic equality
non-convergence fail closed
two-plate exact-q response-cache reuse
serial vs POSIX-fork equality
validation CLI diagnostic-only routing
```

## 后续资格化

要将 adaptive 提升为 production candidate，仍需新增独立证据：

1. 冻结 `ArbitraryQVectorAdaptiveFormalPolicy`，包括误差容差、资源预算、cell batching 和工作负载。
2. 在同一 clean head 上运行 real-hardware adaptive performance preflight。
3. 对 spm/dwave 的 axis、generic、near-diagonal、exact-diagonal 和 rotated q 比较：
   - complete-orbit（可公度点）；
   - fixed periodic N=256/384/512；
   - vector-adaptive tolerance ladder。
4. 对真实 two-plate common-lab logdet 做 tolerance refinement 和独立 adaptive-tree audit。
5. 建立 outer 所需 q-envelope 和 tail convergence。

完成这些证据前，不得开始使用 adaptive 结果作为正式完整 Casimir 积分输入。
