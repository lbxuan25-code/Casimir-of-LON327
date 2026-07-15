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
  first N at the beginning of the accepted convergence window

audit_N
  higher N that confirms the window
```

This early stop changes only workload scheduling. Every evaluated level remains an
independent complete shifted even-N periodic BZ quadrature; no cell or physical
sector receives local refinement.

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
  --logdet-rtol 1e-3 \
  --logdet-atol 1e-14 \
  --output validation/outputs/matsubara/transverse_point_sweet_spot/example.json
```

The summary printed to stdout lists the evaluated N values and the selected
`working_N`/`audit_N` for every requested point. Unresolved points remain explicitly
`not_established`; the command never silently substitutes the highest attempted N
as a qualified result.

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
