# TODO 1 — archive the stopped 0-degree calculation and freeze a diagnostic baseline

## Scope

This TODO freezes the stopped zero-degree SPM campaign and its read-only replay/diagnostic evidence before any material-response or geometry refactor begins.

The historical calculation is permanently classified as:

```text
baseline_id: zero-degree-spm-campaign-3a6260fd6ec8
campaign_id: campaign-3a6260fd6ec8
status: diagnostic_only_unresolved
production_casimir_allowed: false
resume_for_production: false
seed_new_formal_campaign: false
extend_matsubara_range: false
```

The frozen source identity is:

```text
76f990e8dea74038e26f9868f956e2097c4e879d
```

The historical cache, plan, replay output, diagnostic table, or archive must never be discovered, imported, migrated, extended, or reinterpreted by the formal `python -m scripts.full_casimir run` path.

## Final archive state

The stopped formal campaign was inventoried at:

```text
outputs/casimir/production/campaign-3a6260fd6ec8/
```

Its verified source snapshot was:

```text
file_count: 13
bytes: 3447122539
tree_digest: 7edabf45564c167cc855f7a1dae82fdea5327b3896f2d183a01bd1ebb98c56a9
```

The unique retained whole-campaign archive is:

```text
/home/liubx25/casimir-cold-archive/diagnostic_baselines/
  zero-degree-spm-campaign-3a6260fd6ec8/
    campaign-3a6260fd6ec8.tar.gz
```

```text
archive_bytes: 203687112
archive_sha256: 3aa4e652e954c232bfbd47905b741c147665fae88ddd85f147565799d0ca0c7a
restore_verification_passed: true
```

The archive was restored to temporary storage and all 13 files were compared by path, size, mode, and SHA-256 against the planned source snapshot. The temporary restore was removed after verification.

After the archive and retained evidence were verified, the hot formal campaign source was removed from `outputs/casimir/production/`. The only remaining production child is an empty `.locks/` coordination directory.

## Retention model

TODO 1 uses exactly four lifecycle classes:

- `canonical_archive`: one verified retained copy of each unique non-reconstructable raw evidence set;
- `baseline_evidence`: compact evidence required to explain the stopping decision and establish lineage;
- `derived_duplicate`: copied or exactly reconstructable material removable after digest-backed verification;
- `transient_working`: scratch, replay extracts, and intermediate reports removable after their facts are bound into the final manifest.

The retained evidence set includes:

```text
canonical whole-campaign archive and archive manifest
whole-campaign archive plan
archive execution report
restore-verification report
campaign-3a6260fd6ec8_spm_cache_only_joint_u6.json
campaign-3a6260fd6ec8_spm_matched_q.json
campaign-3a6260fd6ec8_spm_microscopic_decay.csv
/home/liubx25/casimir-output-archive-20260721-223949/
prune execution report
post-cleanup audit report
```

The following verified duplicates or transient files were removed:

```text
hot formal campaign source
/home/liubx25/casimir-plans/production_plan_0deg.json
campaign-3a6260fd6ec8_spm_matched_q.csv
campaign-3a6260fd6ec8_matsubara_extract.json
TODO 1 intermediate inventory and replay-analysis reports
```

No retained canonical archive or baseline-evidence item was deleted.

## One-manifest rule

The stopped campaign has one stable generated baseline location:

```text
outputs/casimir/catalog/diagnostic_baselines/
  zero-degree-spm-campaign-3a6260fd6ec8/
    manifest.json
    inventory.tsv
    prune_plan.json
    prune_execution.json
    post_cleanup_audit.json
    formal_campaign_archive/
```

These files are local generated data and are ignored by Git. Re-running baseline bookkeeping updates this same location and must not create timestamped baseline directories.

Final local identities are:

```text
manifest_sha256: fcc3b4b99ae4ba36c2ad5999cfd93c0cf9c50ba5201917f2676ebc4f3a818343
inventory_rows: 27
inventory_sha256: a5c3ff9d9986b2442e17c0c49ee7203700324c33cb0117d18a534531868f6330
prune_plan_sha256: a0b8663744c3424c913b443de55883e36d43f81b817a87136c35039908a810d7
prune_execution_sha256: 9331cff9210505d02b987c797eda412ee9c2158bf9e705a16aa3d67661f1c29b
post_cleanup_audit_sha256: c6d1539fd7f28c5221940ddcb292808a1bcbe84c3d1a364ca3bf702b926ee78c
```

## Frozen numerical conclusion

The exact diagnostic evidence is:

```text
n <= 7:
  all microscopic nodes certified
  partial_free_energy_J_m2: -1.1960685855373808e-07
  finite Matsubara budget: failed
  last-block tail budget: failed

n <= 15:
  all microscopic nodes certified
  partial_free_energy_J_m2: -2.3827370535472416e-07
  block 8..15 / block 4..7: 1.8839607728880299
  finite Matsubara budget: failed
  last-block tail budget: failed

n <= 31:
  one request / 8192 points missing
  status: unresolved
  pairing_status: radial_unresolved
  partial_diagnostic_free_energy_J_m2: -4.3884946002292425e-07

block 16..31 / block 8..15: 1.6902425578441627
formal block-ratio threshold: 0.8
formal Matsubara certificate: false
```

The `n <= 31` value is a partial diagnostic quantity, not a certified finite-domain result. Earlier provisional labels that treated unrelated diagnostic scores as approximately `14.20` and `11.08 nJ/m^2` are rejected and must not be used as free energies.

The frozen conclusion is:

```text
outer-Q finite-domain integration through n <= 15: available as diagnostic evidence
n <= 31 finite-domain closure: incomplete by one request / 8192 points
absolute Matsubara tail: unresolved
formal Matsubara certificate: false
old campaign continuation: forbidden
new n=32..63 extension: not planned
production_casimir_allowed: false
```

## Completion gates

```text
stopped_formal_campaign_inventoried: true
all_known_supporting_sources_inventoried: true
canonical_archives_verified: true
matsubara_failure_bound_to_hashed_evidence: true
historical_status: diagnostic_only_unresolved
formal_campaign_import_path_present: false
loose_duplicate_retention_plan_closed: true
production_casimir_allowed: false
cleanup_execution_completed: true
post_cleanup_audit_completed: true
todo1_complete: true
```

TODO 1 is complete. TODO 2 may begin only as a new formal workflow and may not resume, seed from, or import any object from this diagnostic baseline.
