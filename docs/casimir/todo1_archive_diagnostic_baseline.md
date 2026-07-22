# TODO 1 — archive the stopped 0-degree calculation and freeze a diagnostic baseline

## Scope

This TODO freezes the stopped zero-degree SPM campaign and its read-only replay/diagnostic evidence before any material-response or geometry refactor begins.

The scientific status of the historical calculation is fixed as:

```text
baseline_id: zero-degree-spm-campaign-3a6260fd6ec8
campaign_id: campaign-3a6260fd6ec8
status: diagnostic_only_unresolved
production_casimir_allowed: false
resume_for_production: false
seed_new_formal_campaign: false
extend_matsubara_range: false
```

The frozen source identity of the stopped calculation is:

```text
76f990e8dea74038e26f9868f956e2097c4e879d
```

The current `main` starts from the same tracked source tree, but the historical campaign remains evidence only. Its cache, plan, replay output, or derived tables must never be discovered, imported, migrated, extended, or reinterpreted by `python -m scripts.full_casimir run`.

## Existing local evidence roots

The first inventory must explicitly include the following local roots and files when they exist:

```text
<repository>/outputs/casimir/
/home/liubx25/casimir-output-archive-20260721-223949/
/home/liubx25/casimir-plans/
/home/liubx25/campaign-3a6260fd6ec8_*
```

The loose `campaign-3a6260fd6ec8_*` JSON/CSV exports are diagnostic derivatives. They are not separate campaigns and must not each become a new permanent archive object.

## Separation from the mainline

Historical material is forbidden from all formal campaign locations:

```text
outputs/casimir/production/
outputs/casimir/production/.locks/
```

No historical cache may be copied into a formal campaign cache. No baseline tool may call a microscopic evaluator. No archive or baseline action may change a plan, result, cache, replay, diagnostic, or source file in place.

Git tracks only this contract and compact documentation. Raw caches, archives, logs, plans, replay outputs, and generated baseline manifests remain local generated data.

## Retention model

TODO 1 uses exactly four lifecycle classes.

### `canonical_archive`

Exactly one verified retained copy of non-reconstructable raw evidence. The current preferred root is:

```text
/home/liubx25/casimir-output-archive-20260721-223949/
```

A second byte-identical archive is not retained merely because it has a different file name or location.

### `baseline_evidence`

Compact evidence needed to explain the stopping decision and compare later implementations. This includes the final baseline manifest, key failure metrics, source/plan/cache identities, and hashes of the retained raw evidence.

### `derived_duplicate`

A loose JSON, CSV, report, or copied directory whose information is already present in a retained canonical archive or is exactly reconstructable from retained evidence. It may be removed only after digest/lineage verification and an explicit deletion plan.

### `transient_working`

Temporary replay copies, extracted archives, scratch reports, and one-off analysis outputs. They must not be promoted into the archive merely because they exist. They are removed after the final manifest records the retained evidence they produced.

## One-manifest rule

The stopped campaign has one baseline manifest:

```text
outputs/casimir/catalog/diagnostic_baselines/
  zero-degree-spm-campaign-3a6260fd6ec8/
    manifest.json
    inventory.tsv
```

The manifest is generated local data and is not a formal campaign artifact. Re-running the inventory updates the same baseline location; it must not create timestamped baseline directories.

The manifest must record, for every retained or removable source:

- absolute source path;
- lifecycle class;
- file count and total bytes;
- file SHA-256 or directory tree digest;
- campaign, plan, source-commit, and cache identity when available;
- whether the item is raw, replay, diagnostic report, table, plan, log, or archive;
- canonical retained object or duplicate-of relationship;
- reconstructability and proposed retention action;
- verification state and verification time.

Paths do not determine lifecycle. Classification must be explicit and digest-backed.

## Frozen numerical conclusion

The baseline report must preserve the following conclusion without promoting it to a production result:

```text
outer-Q finite-domain integration: not the current blocker
absolute Matsubara tail: unresolved
old campaign continuation: forbidden
new n=32..63 extension: not planned
```

The recorded diagnostic evidence includes:

```text
n <= 7:  finite-domain joint closure
n <= 15: finite-domain joint closure
n <= 31: one request missing; no formal Matsubara certificate

|F_15| ~= 14.20 nJ/m^2
|F_31| ~= 11.08 nJ/m^2
|F_31| / |F_15| ~= 0.78

block 8..15 / block 4..7   ~= 1.88
block 16..31 / block 8..15 ~= 1.69
formal block-ratio threshold <= 0.8
```

These values are diagnostic evidence from the stopped run. TODO 1 must bind them to their source files and hashes before the baseline is complete.

## Execution phases

### Phase A — read-only inventory

1. Run the existing repository output audit and catalog commands.
2. Inventory the three external `/home/liubx25` evidence groups explicitly.
3. Hash every file and compute deterministic tree digests.
4. Detect byte-identical files and tree-identical directories.
5. Do not move, copy, pack, delete, or modify evidence.

### Phase B — freeze the baseline manifest

1. Select one canonical retained object for each unique raw evidence set.
2. Bind the Matsubara stopping conclusion to exact files and hashes.
3. Mark all historical results `diagnostic_only_unresolved`.
4. Record every loose export as retained evidence, duplicate, reconstructable derivative, or transient working data.
5. Verify that no runtime path in the formal workflow reads the baseline or archive.

### Phase C — bounded cleanup

1. Verify canonical archives by temporary restore and per-file comparison.
2. Build an explicit deletion plan only for `derived_duplicate` and `transient_working` items.
3. Never delete the sole verified copy of non-reconstructable evidence.
4. Keep one manifest and one canonical archive object per unique evidence set.
5. Re-run the output-layout audit and catalog after cleanup.

## Completion gates

TODO 1 is complete only when all of the following are true:

```text
all_known_sources_inventoried: true
canonical_archives_verified: true
matsubara_failure_bound_to_hashed_evidence: true
historical_status: diagnostic_only_unresolved
formal_campaign_import_path_present: false
loose_duplicate_retention_plan_closed: true
production_casimir_allowed: false
```

Until these gates pass, no old source is deleted and TODO 2 does not begin.
