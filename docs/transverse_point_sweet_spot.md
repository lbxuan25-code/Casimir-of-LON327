# Unified transverse-point sweet-spot diagnostic

## Purpose

The repository exposes one public command for selecting the transverse
Brillouin-zone integration budget at fixed external Casimir points:

```bash
python -m validation diagnostic transverse-point-sweet-spot
```

A point is identified by

```text
pairing, q_lab, Matsubara index n, plate angles, separation
```

The command supports both `spm` and `dwave`, multiple external momenta with
independent labels, exact `n=0`, positive Matsubara indices, and repeated complete
periodic-grid shifts.

It does not perform the outer q integral or Matsubara sum. It is diagnostic-only
and cannot by itself authorize production Casimir input.

## Why the budget is point-specific

Different external q magnitudes, q directions, pairings, and Matsubara frequencies
have different transverse-integration difficulty. Applying the hardest point's N to
every point wastes material-grid construction and response time.

The command therefore tracks each `(pairing, q label, n)` independently. Once one
point has enough consecutive accepted N transitions, that frequency is removed
from subsequent levels while unresolved points continue. The JSON records:

```text
working_N
  the lower endpoint of the final accepted adjacent-N transition

audit_N
  the higher endpoint that confirms the transition
```

With the default two consecutive accepted transitions, `N1 -> N2` and `N2 -> N3`
must both pass, and the reported pair is `working_N=N2`, `audit_N=N3`. With the
one-pass smoke setting, the report is `working_N=N_previous`,
`audit_N=N_current`; the two values are never fabricated from the same level.

This early stop changes only workload scheduling. Every evaluated level remains an
independent complete shifted even-N periodic BZ quadrature; no cell or physical
sector receives local refinement.

## CPU parallel policy

The command uses exactly one process-parallel layer and requires one BLAS/OpenMP
thread per process. It never nests a process pool inside another process pool.

At every N level, automatic scheduling compares three safe execution shapes:

```text
q parallelism
  one pairing/shift material cache is built once;
  POSIX-fork workers share the readonly cache copy-on-write;
  q/angle tasks are distributed through the established q evaluator.

material-context parallelism
  independent pairing/shift contexts run in spawn workers;
  each worker builds one material cache and processes q tasks serially;
  simultaneous contexts are capped by the memory budget.

context-wave/q-task parallelism
  the parent builds a memory-safe wave of readonly pairing/shift material caches;
  one POSIX-fork pool inherits all caches in that wave;
  work is flattened into independent `(context, q)` tasks;
  the wave is released before the next set of contexts is built.
```

Automatic mode chooses the shape with the highest process utilization. Ties prefer
q parallelism, then context parallelism, because they retain fewer simultaneous
material contexts. Wave mode is selected when the product of context and q
multiplicities exposes more independent work than either axis alone.

For example, two pairings, three shifts, and three q points expose

```text
6 pairing/shift contexts
18 flattened context-q tasks
```

If memory permits all six contexts in one wave on a 32-CPU host, automatic mode may
use 18 workers rather than being limited to six context workers or three q workers.
At larger N, the same workload can split into multiple memory-capped waves.

The worker and memory controls are:

```text
--workers 0
  use the CPU affinity visible to the process; a positive value is a hard total
  process budget.

--parallel-mode auto|serial|q|context|wave
  automatic selection or an explicit non-nested execution shape.

--memory-budget-gb 0
  use 70% of currently available memory; a positive value sets an explicit budget.

--max-context-workers 0
  automatic memory-limited contexts per wave; a positive value adds a hard cap.

--memory-safety-factor 1.5
  inflate measured/fallback material-context memory before choosing concurrency.
```

The first N level uses a conservative bytes-per-grid-point estimate. Every completed
context then records exact reachable NumPy-array bytes for its material cache. Later
N levels reuse the largest observed bytes-per-point value, so the wave/context cap
adapts to the actual model rather than staying a fixed guess.

Every N-level JSON record contains the selected strategy, worker counts, flattened
task count, contexts per wave, wave count, memory cap, estimated concurrent bytes,
and the reason for the choice. Each wave records exact live cache-array bytes,
worker PIDs, task count, parent cache-build time, and pool wall time.

The output is atomically checkpointed after each completed N level. Automatic resume
is not yet implemented; a partial checkpoint is evidence, not an instruction to
skip unfinished work.

## Acceptance criterion

The primary numerical observable is the actual common-lab two-plate Lifshitz
`logdet` for the requested plate angles and separation.

For one N transition to pass, all requested shifts must satisfy:

```text
operator Ward identity                     hard gate
effective Ward identity                    hard gate
finite/reality/mixing/passivity sheet      hard gate
reflection construction for both plates    hard gate
finite two-plate logdet                     hard gate
adjacent-N logdet tolerance                 convergence gate
cross-shift logdet spread                   convergence gate
```

The exact-static longitudinal residual and the historical strict-static aggregate
remain recorded telemetry. They are not hard gates and receive no q-specific or
pairing-specific exception.

By default two consecutive accepted transitions are required. Thus an established
sweet spot contains at least three N levels and is not based on one accidental
pairwise agreement.

## Example

```bash
env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  OMP_DYNAMIC=FALSE \
  MKL_DYNAMIC=FALSE \
  python -m validation diagnostic transverse-point-sweet-spot \
    --q-point small_axis 0.0100051 0.0 \
    --q-point generic 0.0300152 0.0200101 \
    --q-point diagonal 0.0300152 0.0300152 \
    --pairings spm dwave \
    --matsubara-indices 0 1 2 4 8 \
    --N-candidates 128 192 256 384 512 640 768 \
    --shift 0.5 0.5 \
    --shift 0.25 0.75 \
    --shift 0.75 0.25 \
    --plate-angles-deg 0 17 \
    --required-consecutive-passes 2 \
    --workers 0 \
    --parallel-mode auto \
    --memory-budget-gb 0 \
    --max-context-workers 0 \
    --memory-safety-factor 1.5 \
    --logdet-rtol 1e-3 \
    --logdet-atol 1e-14 \
    --output validation/outputs/matsubara/transverse_point_sweet_spot/example.json
```

The summary printed to stdout lists the parallel plan for every N level and the
selected `working_N`/`audit_N` for every requested point. Unresolved points remain
explicitly `not_established`; the command never silently substitutes the highest
attempted N as a qualified result.

## Public-surface boundary

The former public single-point convergence routes were removed:

```text
matsubara positive-point
static nk-scan
diagnostic arbitrary-q-uniform-refinement
```

Formal arbitrary-q qualification, performance checks, quadrature-method comparison,
and complete outer-integration commands have different responsibilities and remain
separate. They do not constitute alternative public single-point sweet-spot tools.
