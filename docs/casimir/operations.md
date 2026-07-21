# Operations

## 1. Create an immutable plan

All formal work begins at the unified command:

```bash
python -m scripts.full_casimir plan \
  --pairings spm dwave \
  --temperature-K 10 \
  --distances-nm 20 \
  --angles-deg 0 \
  --plan-output production_plan.json
```

Review the printed scientific policy, case matrix, Git commit and `plan_sha256`.
Changing any scientific input requires a new plan and therefore a new scientific
identity.

## 2. Fresh execution

A new campaign must start from an empty formal campaign directory:

```bash
python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 <PLAN_SHA256> \
  --fresh
```

`--fresh` refuses an existing campaign. Historical caches are never discovered,
imported, migrated or extended into the campaign.

## 3. Resume

Continue the same campaign with the same plan and source identity:

```bash
python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 <PLAN_SHA256> \
  --resume
```

Resume permits execution-only changes such as worker count and memory budget, but
rejects changes to physical inputs, numerical policy, certificates or scientific
identity. It reuses only exact certified `(pairing,n,qx.hex,qy.hex)` cache entries.

## 4. Artifact reading order

For each case under `production/<campaign-id>/runs/<physical-case>/`:

1. `identity.json`: physical case and campaign binding;
2. `manifest.json`: attempt state and production authorization;
3. `summary.json`: selected cutoffs, errors and termination reason;
4. `result.json`: complete layered numerical evidence;
5. `cache/identity.json`: certified-cache identity;
6. `cache/certified_points.json`: resume data, not a human result summary.

A case is `completed` only when `production_casimir_allowed=true`. Numerical output
without full policy and error-budget closure remains `diagnostic_only` or
`unresolved`.

## 5. Historical evidence before output cleanup

Old three-shift histories may be audited without starting calculation:

```bash
python -m scripts.full_casimir shift-audit \
  --input <old-certified-points.json> \
  --output <two-shift-replay.json>
```

The report is evidence only and can never seed the formal campaign.

## 6. First fresh 0-degree campaign

The first full calculation after TODO 8 is the local fresh `T=10 K`, `d=20 nm`,
`theta=0°`, SPM+d-wave qualification campaign. It begins only after old outputs are
archived or removed and the formal output root is empty. The identical campaign may
then be transferred to the server and continued under the same Git commit and plan
SHA.
