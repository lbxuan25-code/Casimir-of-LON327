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
imported, migrated or extended into the campaign. Before the campaign directory is
created, the command acquires one campaign-owner lock under
`production/.locks/`; a second process for the same campaign is rejected.

## 3. Resume and checkpoint recovery

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
Before resuming, a read-only recovery preflight is written to
`reports/recovery.json`; the authoritative checkpoint is the last atomically committed
`cache/certified_points.json`. Orphan `*.tmp` files are reported but never promoted.

A dead process leaves its immutable owner record and token-specific heartbeat. A stale
owner can be replaced only by an explicit resume command:

```bash
python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 <PLAN_SHA256> \
  --resume \
  --take-over-stale-lock
```

The old owner and heartbeat are archived under `production/.locks/history/`. A live
local PID is never taken over, even when its heartbeat is old. The defaults are a
30-second lock heartbeat and a 300-second stale threshold.

## 4. Bounded engineering retries

Automatic retry is disabled by default. A finite retry allowance may be added without
changing the scientific identity:

```bash
python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 <PLAN_SHA256> \
  --resume \
  --max-engineering-retries 2 \
  --retry-delay-seconds 30
```

Each retry re-enters through formal resume, reuses atomic caches, skips already
completed cases, and remains under the same campaign lock. Numerical `unresolved` or
`diagnostic_only` results are not engineering failures and are not silently converted
into authorized results.

## 5. Progress and artifact reading order

Monitor without changing the run:

```bash
python -m scripts.full_casimir status --campaign <campaign-id>
python -m scripts.full_casimir status --campaign <campaign-id> --watch
```

For each case under `production/<campaign-id>/runs/<physical-case>/`:

1. `identity.json`: physical case and campaign binding;
2. `manifest.json`: attempt state and production authorization;
3. `summary.json`: selected cutoffs, errors and termination reason;
4. `result.json`: complete layered numerical evidence;
5. `cache/identity.json`: certified-cache identity;
6. `cache/certified_points.json`: atomic resume data, not a human result summary;
7. `progress.json` and `progress.events.jsonl`: non-authoritative observation data.

A case is `completed` only when `production_casimir_allowed=true`. Numerical output
without full policy and error-budget closure remains `diagnostic_only` or
`unresolved`.

## 6. Reproducibility and source proof

After every completed run attempt, the guarded command writes:

```text
reports/source_proof.json
reports/artifact_manifest.json
reports/reproducibility.json
```

The source proof records the exact Git commit and tree plus SHA-256 hashes for every
tracked source file. The artifact manifest hashes the registered plan, policy,
identities, configurations, manifests, results and certified caches. The
reproducibility record links both digests to the execution environment and resources;
execution details remain separate from the scientific-policy SHA.

Verify the bundle without starting calculation:

```bash
python -m scripts.full_casimir proof --campaign <campaign-id>
```

By default this verifies the registered plan, all recorded authoritative artifact
hashes, and the current checkout against the recorded source commit/tree/file-set.
Use `--skip-current-source-check` only when verifying transferred artifacts on a
machine that does not have the original source checkout.

## 7. Historical evidence before output cleanup

Old three-shift histories may be audited without starting calculation:

```bash
python -m scripts.full_casimir shift-audit \
  --input <old-certified-points.json> \
  --output <two-shift-replay.json>
```

The report is evidence only and can never seed the formal campaign.

## 8. First fresh 0-degree campaign

With TODO items 7 and 8 implemented, the workflow has the required ownership,
recovery, progress and reproducibility functions for the first local fresh `T=10 K`,
`d=20 nm`, `theta=0°`, SPM+d-wave qualification campaign. Running that expensive
qualification and using it to reduce the conservative outer-Q/Matsubara ceilings are
separate execution tasks, not part of the implementation change itself.
