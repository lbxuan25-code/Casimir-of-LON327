# Casimir outer integration

This directory intentionally contains no executable full outer-integration pipeline yet.

The retired monolithic `finite_q_bdg_casimir_pipeline.py` encoded superseded zero-RHS Ward, positive-frequency-to-zero extrapolation and legacy TE/TM conventions. Do not restore or copy it.

## Microscopic intake implementation

The typed library contains `ArbitraryQPeriodicBZContract-v3`:

```text
fixed shifted even-N, N x N full periodic BZ lattice
exact q_crystal = R(-theta) q_lab
no q rounding, wrapping or interpolation
one q-workspace implementation shared with the commensurate reference
operator diagnostics reuse existing shifted Hamiltonians and vertices
readonly MaterialGridCache-v3
separate material-state and grid fingerprints
CrystalResponseCache-v3 with complete numerical-policy identity
runtime-sized eigensystem/vertex/Kubo batches
canonical deterministic reduction blocks
exact zero + positive Matsubara shared q workspace
q_lab + angle-batch persistent POSIX-fork execution
actual child BLAS threadpool verification
```

The complete-orbit backend remains the commensurate-q reference and regression authority.

Authoritative contracts:

```text
docs/full_outer_integration_handoff.md
docs/arbitrary_q_periodic_bz_design.md
validation/README.md
```

## Before a formal run

Use a clean checkout of the exact branch head:

```bash
git status --porcelain --untracked-files=all
```

The command must print nothing. Formal evidence records the commit, tree object, tracked index and source-tree fingerprint. Any source edit after performance preflight invalidates the manifest.

Set numerical thread limits **before Python starts**:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
```

Install the validation dependencies:

```bash
python -m pip install -e ".[dev]"
```

Both parent and child processes verify actual runtime threadpools with `threadpoolctl`; environment strings alone do not establish a pass.

## Mandatory formal order

### 1. Target-machine performance preflight

```bash
python -m validation matsubara arbitrary-q-performance-preflight \
  --pairings spm dwave \
  --N 128 \
  --q-tasks 8 \
  --workers 8 \
  --matsubara-indices 0 1 2 4 8 \
  --canonical-block-size 4096 \
  --runtime-chunk-sizes 4096 16384 \
  --temperature-K 10 \
  --delta0-eV 0.1 \
  --eta-eV 1e-8 \
  --comparison-atol 2e-12 \
  --comparison-rtol 2e-11 \
  --minimum-speedup 4 \
  --minimum-cpu-wall-ratio 4 \
  --maximum-pool-overhead-fraction 0.05 \
  --output validation/outputs/matsubara/arbitrary_q_performance_preflight/real_head.json
```

`ArbitraryQPerformanceWorkloadV2` measures three workloads:

```text
outer_q_batch_v2: 8 tasks / 8 workers
qualification_primary_v2: 4 tasks / 4 workers
qualification_audit_v2: 1 task / 1 worker
```

The manifest also proves that runtime 16384 creates one q workspace per 16384 points and reduces batched eigensystem calls relative to runtime 4096 while preserving the canonical packed result. Pool shutdown is measured only after close/join.

### 2. Do not modify source

Re-run:

```bash
git status --porcelain --untracked-files=all
```

It must still be empty. Do not edit, commit, rebase, install generated source files, or create unignored files between the two formal commands.

### 3. Public formal numerical gate

```bash
python -m validation matsubara arbitrary-q-periodic-bz-qualification \
  --performance-manifest \
    validation/outputs/matsubara/arbitrary_q_performance_preflight/real_head.json \
  --pairings spm dwave \
  --N-values 256 384 512 \
  --reference-nk 1256 \
  --reference-order 384 \
  --reference-panel-count 16 \
  --reference-workers 8 \
  --reference-task-size 4 \
  --workers 4 \
  --matsubara-indices 0 1 8 \
  --canonical-block-size 4096 \
  --runtime-chunk-size 16384 \
  --temperature-K 10 \
  --delta0-eV 0.1 \
  --eta-eV 1e-8 \
  --separation-nm 20 \
  --primitive-tolerance 1e-3 \
  --primitive-atol 1e-12 \
  --reflection-tolerance 3e-4 \
  --reflection-atol 1e-12 \
  --logdet-tolerance 3e-4 \
  --logdet-atol 1e-14 \
  --diagonal-observable-tolerance 1e-3 \
  --diagonal-observable-atol 1e-12 \
  --ward-tolerance 1e-7 \
  --ward-absolute-tolerance 1e-12
```

`ArbitraryQFormalPolicyV2` includes every absolute/relative comparison tolerance, Ward tolerance, physical work point, reference panel/workload parameter and execution policy. Looser or changed values can run only with `--diagnostic-nonformal` and can never authorize outer integration.

The numerical core itself writes only:

```text
diagnostic_result_passed
diagnostic_result_failed
```

The public clean-source gate alone may promote a passed result to:

```text
qualified_for_diagnostic_outer_integration
```

## What the large-N gate proves

For each pairing it builds:

```text
primary N=256
primary N=384
primary N=512
audit A=(1/4,3/4) at N=512
audit B=(3/4,1/4) at N=512
```

The primary contexts use four workers for four q tasks. Audit contexts use one worker for one task.

Paired shift results are formed at the packed-primitive level for each plate. The final consumed two-plate observable is directly gated at every Matsubara index:

```text
plate 1 theta = 0 degrees
plate 2 theta = 17 degrees
common lab LT basis
two-plate logdet at N=256,384,512
N refinement
audit A/B two-plate spread
paired-plate two-plate logdet
primary N512 vs paired sensitivity
```

Single-plate convergence cannot substitute for this nonlinear final-observable gate.

## Momentum boundary

The code supports, without wrapping:

```text
|q_x| <= pi
|q_y| <= pi
```

This is a syntactic principal-domain boundary, not a numerically qualified outer envelope. The qualification manifest deliberately reports:

```text
qualified_outer_q_envelope_established = False
continuous_angle_coverage_established = False
outer_tail_requirement_bound = False
```

A future outer configuration must determine the q range required by its separation interval, angle range and tail tolerance, then establish a separate envelope manifest before evaluating outer nodes.

## Future full outer layer

Only after the microscopic manifests and a later q-envelope contract pass may the outer layer be implemented. It must remain thin orchestration over the typed library and provide:

- exact zero-Matsubara density/stiffness sheet response;
- positive-Matsubara conductivity sheet response;
- common lab LT tangential-electric reflections;
- `lno327.casimir.lifshitz_integrand.passive_sheet_logdet`;
- explicit Matsubara prime weight;
- q/angle quadrature and tail convergence;
- restartable caches and compact worker-side logdet payloads;
- energy convergence before torque differentiation.

Current state:

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```
