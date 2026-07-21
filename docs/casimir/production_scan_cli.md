# Unified production scan command

The stable orchestration entry for full-Casimir physical case matrices is:

```bash
python -m scripts.full_casimir
```

The underlying scientific API remains `lno327.casimir`. This command layer only plans,
identifies, and dispatches one or more physical cases through that API.

## Safe planning

Before running any expensive work, resolve the exact case matrix:

```bash
python -m scripts.full_casimir plan \
  --pairings spm dwave \
  --distances-nm 10 20 40 \
  --angles-deg 0 45 90 \
  --profile candidate-policy
```

The same plan can be written as JSON:

```bash
python -m scripts.full_casimir plan \
  --pairings spm dwave \
  --distances-nm 20 \
  --angles-deg 0 2 4 \
  --profile candidate-policy \
  --plan-output outputs/casimir/catalog/candidate-scan-plan.json
```

`plan` never executes microscopic or outer-integral work.

## Single case

A single angle and distance use the same interface as a scan:

```bash
python -m scripts.full_casimir run \
  --pairings dwave \
  --distances-nm 20 \
  --angles-deg 0 \
  --profile candidate-policy
```

## Explicit multi-case scan

```bash
python -m scripts.full_casimir run \
  --pairings spm dwave \
  --distances-nm 10 20 40 \
  --angles-deg 0 15 30 45 60 75 90 \
  --profile candidate-policy
```

## Inclusive range syntax

Angles and distances may instead be generated from exactly divisible inclusive ranges:

```bash
python -m scripts.full_casimir plan \
  --distance-min-nm 10 \
  --distance-max-nm 60 \
  --distance-step-nm 5 \
  --angle-min-deg 0 \
  --angle-max-deg 90 \
  --angle-step-deg 2
```

For each axis, explicit values and range syntax are mutually exclusive. A range requires
all three of min, max, and step. Decimal arithmetic is used to avoid accumulated floating
point drift.

With no physical-grid arguments, the safe default is one case at `d=20 nm` and `theta=0`.

## Physical case identity

Case names now encode the requested temperature, separation, angle, pairing, and the
transitional profile label. Examples:

```text
spm_T10K_d20nm_theta_p000deg_candidate-policy
dwave_T12p5K_d17p25nm_theta_p002p5deg_candidate-policy
```

The default legacy identity remains byte-for-byte compatible with existing 10 K, 20 nm,
integer-angle runs.

The profile field is transitional. The cache-management task will replace human version
labels with a policy/code identity contract. Until that task is complete, changing any
physical or numerical option requires a distinct profile and run directory.

## Other commands

The same top-level entry routes the maintained analysis and data tools:

```bash
python -m scripts.full_casimir resources
python -m scripts.full_casimir diagnose --help
python -m scripts.full_casimir audit --help
python -m scripts.full_casimir data --help
python -m scripts.full_casimir qualification --help
```

The old `scripts.full_casimir.workflow` surface remains available only through:

```bash
python -m scripts.full_casimir legacy-workflow --help
```

Direct module invocations remain temporarily compatible, but new runbooks should use the
single top-level entry.

## Current authorization boundary

This interface does not authorize formal production. New formal scans must wait until the
error budget and numerical policy are frozen from the current benchmark. Until then,
`plan` and software dry-runs are allowed; expensive `run` commands are diagnostic only.
