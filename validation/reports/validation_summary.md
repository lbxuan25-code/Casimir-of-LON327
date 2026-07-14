# Two-band finite-q validation summary

## Current status

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

The retained transverse method is one full-period equal-panel composite
Gauss-Legendre rule. Per-point adaptation changes only total order and point budget;
it does not switch quadrature method or apply even/C4/axis/diagonal/q-direction
symmetry reductions.

## Total Matsubara validation

The common spm/d-wave backend evaluates true Matsubara `n=0` and all requested
positive frequencies in one complete-orbit microscopic batch. Eigensystems are shared
across frequency. Postprocessing branches only after the primitive integral:

```text
n = 0  -> exact divided difference -> static density/stiffness -> static R/logdet
n > 0  -> positive-frequency Kubo -> conductivity -> positive R/logdet
```

Zero frequency is never obtained from `sigma=-K/xi`.

## Acceptance policy

Hard at every point:

- RHS-aware Ward and physical pipeline;
- exact-static divided difference for `n=0`;
- strict static Ward gate;
- sheet validation, reflection construction and signed-real logdet;
- reflection/logdet drift `<= 1e-3`.

Numerical response acceptance:

```text
n=0 primary response: strict <= 1e-3, soft <= 2e-3
n>0 sigma:            strict <= 1e-3, soft <= 2e-3
```

Static soft does not soften any static physics contract. Any soft case requires
non-worsening static and positive drift across successive stage pairs or the final hard
pair, followed by a shifted-periodic-cut audit.

Default stage pairs:

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
```

## Difficult positive-frequency evidence

For d-wave `nk=1256`, `(mx,my)=(1,1)`, positive indices `1,2`:

```text
C256-C224 sigma relative: 7.375e-3, 5.773e-3
C384-C320 sigma relative: 1.668e-3, 1.362e-3
C384-C320 R relative:     6.012e-6, 4.751e-6
C384-C320 logdet relative:1.655e-5, 1.302e-5
Ward / physical pipeline: pass
```

This does not establish a strict conductivity reference, but it satisfies the current
soft tensor target and shows much stronger observable stability.

## Performance evidence

Real `nk=1256`, `(1,1)`, `C64`, combined `n=(0,1,2)`:

```text
spm:   serial 20.709 s, 8-process 3.149 s, speedup 6.577x, CPU/wall 7.469
dwave: serial 20.655 s, 8-process 3.055 s, speedup 6.761x, CPU/wall 7.751
callbacks = 64 for three frequencies; serial/process exact equality = True
```

Independent d-wave exact-static agreement:

```text
maximum component mixed ratio = 1.600e-5
maximum RHS mixed ratio       = 2.237e-5
passed                         = True
```

The timing breakdown remains dominated by batched material workspace (~84%) and
batched q workspace (~13%); Kubo factors, contractions and primitive packing together
are ~3%. Process startup and postprocessing are about 1% of the C64 wall time.

## Verification

GitHub Actions run `29306734394` passed completely on head
`fba40ce7ef30a3ea7fa3084e56de2e1a34518699`, including targeted contracts, the full
repository suite, CLI routing, fork-process fixed-Gauss smoke, the blocking
preflight-to-manifest-to-total-scan subprocess chain, static-soft schema v2, and legacy
entry points.

A formal local scan must regenerate the preflight manifest after pulling the current
head, because manifests are intentionally bound to the exact Git head and runtime
controls.

## Artifact and maintenance policy

Raw CSV, JSON, txt, log, figures, arrays and intermediate validation outputs are local
reproducible artifacts and are ignored by Git. Keep the PR draft and unmerged; do not
claim final Casimir readiness before the total outer integration and energy/torque
convergence reports are complete.
