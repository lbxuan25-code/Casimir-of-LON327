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

## Universal convergence policy

The convergence definition is identical for every pairing, q magnitude, q
direction, and Matsubara index. There is no axis, diagonal, near-diagonal, zero-mode,
positive-frequency, `spm`, or `dwave` exception.

Every logdet comparison is evaluated in this order:

```text
1. absolute error <= logdet_atol
2. otherwise relative error <= logdet_rtol
3. otherwise fail
```

The default provisional global tolerances are

```text
logdet_atol = 1e-6
logdet_rtol = 1e-3
```

The absolute tolerance is a universal numerical floor, not a special allowance for
one difficult point. Its present value is provisional until the complete outer-q,
Matsubara, and torque error budget is available. Users may override both tolerances
explicitly, but one run applies the chosen values uniformly to every requested point.

Physical closure remains a hard gate and is never bypassed by an absolute or
relative numerical tolerance.

## Two universal establishment routes

A point can establish a working/audit pair through either of two routes.

### Strict consecutive-adjacent route

For one adjacent-N transition to pass, all requested shifts must satisfy:

```text
operator Ward identity                     hard gate
effective Ward identity                    hard gate
finite/reality/mixing/passivity sheet      hard gate
reflection construction for both plates    hard gate
finite two-plate logdet                     hard gate
adjacent-N logdet absolute-or-relative gate convergence gate
cross-shift absolute-or-relative spread     convergence gate
```

By default two consecutive accepted transitions are required. Thus
`N1 -> N2` and `N2 -> N3` must both pass, and the reported pair is
`working_N=N2`, `audit_N=N3`.

### Three-level oscillatory-envelope route

Periodic full-grid quadrature can approach its limit through small non-monotone
aliasing oscillations. A point may therefore also establish when the most recent
three complete N levels satisfy all of the following:

```text
all hard physical gates pass at every level
cross-shift spread passes at every level
one joint envelope over all three N levels and all shifts passes
```

The joint envelope is

```text
max(logdet over N and shift) - min(logdet over N and shift)
```

and uses the same universal absolute-first, relative-fallback tolerances. No point
receives a custom envelope width. When this route passes, the report uses the final
two levels as `working_N` and `audit_N` and records the complete three-level N window.

The output field `establishment_mode` is one of

```text
strict_consecutive_adjacent
three_level_oscillatory_envelope
```

The exact-static longitudinal residual and the historical strict-static aggregate
remain recorded telemetry. They are not hard gates and receive no q-specific or
pairing-specific exception.

## Why the budget remains point-specific

Different external q magnitudes, q directions, pairings, and Matsubara frequencies
have different transverse-integration difficulty. Applying the hardest point's N to
every point wastes material-grid construction and response time.

The command therefore tracks each `(pairing, q label, n)` independently. Once one
point passes either universal establishment route, that frequency is removed from
subsequent levels while unresolved points continue. The JSON records:

```text
working_N
  lower endpoint of the final accepted working/audit pair

audit_N
  higher endpoint that confirms the accepted convergence window

establishment_mode
  strict consecutive-adjacent or three-level oscillatory envelope
```

This early stop changes only workload scheduling. Every evaluated level remains an
independent complete shifted even-N periodic BZ quadrature; no cell or physical
sector receives local refinement.

## CPU parallel policy

The command uses exactly one process-parallel layer and requires one BLAS/OpenMP
thread per process. It never nests a process pool inside another process pool.

At every N level, automatic scheduling compares four execution shapes:

```text
serial
  one material context and one q task at a time.

q parallelism
  one pairing/shift material cache is built once;
  POSIX-fork workers share the readonly cache copy-on-write;
  q/angle tasks from one identical-frequency group are distributed in parallel.

material-context parallelism
  independent pairing/shift contexts run in spawn workers;
  each worker builds one material cache and processes its q groups serially;
  simultaneous contexts are capped by the memory budget.

context-wave/q parallelism
  a memory-safe set of pairing/shift caches is built in the parent;
  one POSIX-fork pool inherits all readonly caches in the wave;
  all `(context, q)` tasks from every active Matsubara group are flattened into
  the same dynamic work queue.
```

Wave planning counts every flattened task in a context, including q labels whose
active Matsubara sets differ and therefore belong to different frequency groups.
Automatic mode chooses the shape that can occupy more CPU processes. A tie prefers
the simpler/smaller-memory shape in the order q, context, wave.

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
  automatic memory-limited context count; a positive value adds a hard cap.

--memory-safety-factor 1.5
  inflate measured/fallback material-context memory before choosing concurrency.
```

The first N level uses a conservative bytes-per-grid-point estimate. Every completed
context then records exact reachable NumPy-array bytes for its material cache. Later
N levels reuse the largest observed bytes-per-point value, so the context-worker cap
adapts to the actual model rather than staying a fixed guess.

Every N-level JSON record contains the selected strategy, worker counts, memory cap,
estimated concurrent bytes and the reason for the choice. The output is atomically
checkpointed after each completed N level. Automatic resume is not yet implemented;
a partial checkpoint is evidence, not an instruction to skip unfinished work.

## Output contract

The v4 JSON records

```text
schema = transverse-point-sweet-spot-v4
convergence_policy.scope = universal for all requested points
convergence_policy.comparison_order = absolute first, relative fallback
convergence_policy.q_or_frequency_specific_exceptions = false
```

Each comparison records

```text
absolute
relative
absolute_tolerance
relative_tolerance
absolute_passed
relative_passed
passed_by = absolute | relative | failed
passed
```

Each history level also contains the current three-level oscillatory-envelope
assessment when enough levels are available.

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
    --logdet-rtol 1e-3 \
    --logdet-atol 1e-6 \
    --output validation/outputs/matsubara/transverse_point_sweet_spot/example.json
```

The summary printed to stdout lists the parallel plan and the selected
`working_N`/`audit_N`/`establishment_mode` for every requested point. Unresolved
points remain explicitly `not_established`; the command never silently substitutes
the highest attempted N as a qualified result.

## Public-surface boundary

The former public single-point convergence routes were removed:

```text
matsubara positive-point
static nk-scan
diagnostic arbitrary-q-uniform-refinement
```

The large numerical implementation behind the public command is retained only as an
internal library engine. Formal arbitrary-q qualification, performance checks,
quadrature-method comparison, and complete outer-integration commands have different
responsibilities and remain separate. They do not constitute alternative public
single-point sweet-spot tools.
