# Arbitrary-q uniform-refinement diagnostic

## Scope

This diagnostic evaluates exact arbitrary crystal momentum on a sequence of fixed,
shifted, even-`N` full-period Brillouin-zone grids. Every level is an independent
complete periodic quadrature. No region, cell, node family or physical sector is
assigned a different spatial resolution.

The command is diagnostic-only:

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

## Command

```bash
python -m validation diagnostic arbitrary-q-uniform-refinement \
  --pairing dwave \
  --q-model 0.0300152 0.0200101 \
  --matsubara-indices 0 1 \
  --N-values 64 96 128 192 \
  --shift 0.5 0.5 \
  --canonical-block 4096 \
  --runtime-chunk 16384 \
  --primitive-rtol 1e-3 \
  --primitive-atol 1e-12 \
  --ward-tolerance 1e-7 \
  --ward-absolute-tolerance 1e-12 \
  --output validation/outputs/matsubara/arbitrary_q_uniform_refinement/dwave.json
```

## Recorded convergence data

Adjacent `N` levels are compared before nonlinear postprocessing using the complete
packed primitive vector. The report resolves changes into:

```text
direct/contact
collective counterterm
phase-direct terms
Ward RHS

for every requested Matsubara frequency:
  electromagnetic bubble
  collective bubble
  electromagnetic-to-collective block
  collective-to-electromagnetic block
```

Each block records:

```text
previous maximum absolute norm
current maximum absolute norm
maximum absolute change
relative change
mixed absolute-relative threshold
mixed ratio
pass/fail
```

The report names the worst block at every adjacent pair of levels.

## Uniform zero-mode physical gate

Every zero-Matsubara integration point uses the same gate. There are no q-direction,
near-diagonal or pairing-specific exemptions.

Hard requirements are:

```text
crystal-xy effective Ward validation passed
finite static kernel and extracted sheet channels
reality tolerance passed
density-transverse mixing tolerance passed
static passivity passed
reflection constructed without a nonphysical pole
passive zero-mode logdet constructed
```

The local-LT longitudinal residual is always evaluated and retained, but is not a
hard gate:

```text
relative_longitudinal_gauge_residual:
  recorded at every point
  compared with the configured diagnostic tolerance
  warning emitted when above tolerance
  never by itself blocks sheet, reflection or logdet construction
```

The historical strict-static aggregate remains telemetry only. Casimir-oriented
convergence is decided from physical closure plus the quantities actually consumed
by the outer calculation: static reflection, zero-mode logdet and their `N`/shift
stability.

## Physical telemetry

Every complete grid is unpacked once and passed through the established physical
pipeline. For each Matsubara frequency the JSON records:

```text
integrated Ward pass and mixed ratio
Schur condition number
historical strict-static diagnostic where applicable
static longitudinal residual/tolerance/warning where applicable
sheet validation
reflection construction
primary-response norm
reflection norm
logdet
chi_bar and dbar_t where applicable
error and warning text
```

Adjacent levels also compare primary response, reflection and logdet. This separates
microscopic primitive convergence from downstream observable sensitivity. A
nonobservable primitive defect does not reject a Casimir result when all hard
physical gates pass and the final reflection/logdet observables are stable.

## Retained numerical boundary

The diagnostic uses only:

```text
fixed even-N full periodic BZ grids
exact q without rounding or wrapping
one established q-workspace and primitive kernel
all requested Matsubara frequencies sharing each q workspace
full primitive integration before phase-Hessian and Schur processing
```

Formal arbitrary-q qualification remains the clean-source periodic-BZ gate. This
command supplies detailed numerical evidence but cannot authorize outer Casimir
input by itself.
