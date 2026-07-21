# Unified production scan command

The stable orchestration entry for full-Casimir physical case matrices is:

```bash
python -m scripts.full_casimir
```

The scientific API remains `lno327.casimir`. The orchestration layer freezes a
scientific policy and case matrix, creates a production campaign identity, and
dispatches cases through the canonical API.

## Production identity model

Formal data are organized as:

```text
outputs/casimir/production/<campaign-id>/
├── campaign.json
├── policy.json
├── plans/
├── runs/
└── reports/
```

A campaign identity is derived from:

```text
scientific policy + exact Git commit + production contract version
```

Worker count, CPU allocation, parallel mode, memory budget and certifier batch
size are execution settings. They may change between resume attempts and do not
change the scientific cache identity.

A physical case is one pairing, temperature, separation and angle. Each case has
an independent certified-point cache and two fail-closed sidecars:

```text
runs/<physical-case>/identity.json
runs/<physical-case>/cache/identity.json
```

The formal path never scans, imports, migrates or extends legacy profile caches.

## 1. Freeze a plan

Before running expensive work, create a plan from a clean tracked worktree:

```bash
python -m scripts.full_casimir plan \
  --pairings spm dwave \
  --distances-nm 10 20 40 \
  --angles-deg 0 45 90 \
  --N-candidates 128 192 256 384 512 640 768 896 \
  --logdet-rtol 0.002 \
  --plan-output production_plan.json
```

The command prints and stores:

- `campaign_id` and full campaign SHA;
- scientific-policy SHA;
- Git commit;
- every physical case identity;
- plan SHA.

`plan` never performs microscopic or outer-integral work. Human development
labels such as `v2`, `v3` or `candidate-policy` are not part of formal names.

Angles and distances can also use exactly divisible inclusive ranges:

```bash
python -m scripts.full_casimir plan \
  --distance-min-nm 10 \
  --distance-max-nm 60 \
  --distance-step-nm 5 \
  --angle-min-deg 0 \
  --angle-max-deg 90 \
  --angle-step-deg 2 \
  --plan-output production_plan.json
```

For each axis, explicit values and range syntax are mutually exclusive.

## 2. Start from empty caches

A new server production campaign must use `--fresh`:

```bash
PLAN_SHA="$(python - <<'PY'
import json
print(json.load(open('production_plan.json'))['plan_sha256'])
PY
)"

python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 "$PLAN_SHA" \
  --fresh \
  --campaign-root outputs/casimir/production \
  --worker-cap 32 \
  --memory-budget-gb 64 \
  --parallel-mode q
```

`--fresh` requires the campaign directory not to exist. It never overwrites or
reuses another output directory.

## 3. Resume the same campaign

After interruption, use the same SHA-confirmed plan:

```bash
python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 "$PLAN_SHA" \
  --resume \
  --campaign-root outputs/casimir/production \
  --worker-cap 24 \
  --memory-budget-gb 48 \
  --parallel-mode context
```

Changing execution resources is allowed. Any physical parameter, numerical
acceptance rule, ladder, error target, Git commit or identity schema mismatch is
rejected.

A later plan may add cases to the same campaign when its policy and Git identity
are unchanged. Existing completed cases are retained; new planned cases start
from empty independent caches.

## State rules

```text
fresh + missing campaign       -> create
fresh + existing campaign      -> reject
resume + missing campaign      -> reject
resume + matching campaign     -> continue/register matching plan
resume + scientific mismatch   -> reject
case directory without formal identity sidecars -> reject
```

There is no formal `--overwrite`, force-reuse or ignore-policy-mismatch option.

## Legacy boundary

Historical `runtime_budget_v3`, `0deg_pilot_v*` and qualification data remain
available to old diagnostic and qualification routes. They are benchmark
evidence only and cannot be discovered automatically by the production command.

## Current authorization boundary

The campaign and cache contracts are implemented, but the final production
policy is not yet authorized. Formal server runs must wait until the current
outer-integral benchmark closes the error budget and freezes the economical
numerical policy.
