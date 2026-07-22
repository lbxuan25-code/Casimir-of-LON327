# TODO 1 — verified zero-degree baseline findings

This note records the final evidence and retention outcome for the stopped formal SPM campaign. It does not authorize production use, resume, or reuse as a formal campaign seed.

## Identity

```text
baseline_id: zero-degree-spm-campaign-3a6260fd6ec8
campaign_id: campaign-3a6260fd6ec8
source_commit: 76f990e8dea74038e26f9868f956e2097c4e879d
execution_state: interrupted
scientific_state: diagnostic_only_unresolved
production_casimir_allowed: false
resume_for_production: false
```

The historical run manifest said `running`, but there was no active campaign lock and no final `result.json`, `summary.json`, `source_proof.json`, `artifact_manifest.json`, or `reproducibility.json`. The campaign is therefore an interrupted diagnostic run with an atomically retained cache, not a finalized formal result.

## Verified formal source snapshot

The source campaign was originally located at:

```text
outputs/casimir/production/campaign-3a6260fd6ec8/
```

```text
bytes: 3447122539
file_count: 13
tree_digest: 7edabf45564c167cc855f7a1dae82fdea5327b3896f2d183a01bd1ebb98c56a9
```

Authoritative cache:

```text
runs/spm_T10K_d20nm_theta_p000deg/cache/certified_points.json
bytes: 3443930560
sha256: b42a04b65f48419f21605f5b7098f8b66e3da385bb8c0ee3145e715f7280382c
processed_entries: 258560
invalid_entries: 0
unique_q_count: 13568
```

Formal plan:

```text
plans/c28f4a066ed474339ed269585960cae676e24ff61bde4f19580462b5dec2cbd4.json
sha256: b85a444f41287d0ed969ef7cd22afa0d3501d10377acd7341d3af4d6acfe2c50
```

The loose external `production_plan_0deg.json` was byte-identical to the formal plan.

## Canonical archive and restore verification

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
restored_file_count: 13
restored_total_bytes: 3447122539
```

The archive was restored into temporary storage and every file was compared against the approved source plan by relative path, size, mode, and SHA-256. The temporary restore directory was then removed.

## Cache-only finite-domain replay

### Cutoff `n <= 7`

```text
status: adaptive_finite_partial
all_microscopic_nodes_certified: true
missing_request_count: 0
missing_point_count: 0
partial_free_energy_J_m2: -1.1960685855373808e-07
finite_matsubara_budget_passed: false
last_block_tail_budget_passed: false
```

### Cutoff `n <= 15`

```text
status: adaptive_finite_partial
all_microscopic_nodes_certified: true
missing_request_count: 0
missing_point_count: 0
partial_free_energy_J_m2: -2.3827370535472416e-07
finite_matsubara_budget_passed: false
last_block_tail_budget_passed: false
block_ratio_8_15_over_4_7: 1.8839607728880299
formal_ratio_threshold: 0.8
```

### Cutoff `n <= 31`

```text
status: unresolved
pairing_status: radial_unresolved
all_microscopic_nodes_certified: false
missing_request_count: 1
missing_point_count: 8192
partial_diagnostic_free_energy_J_m2: -4.3884946002292425e-07
```

The `n <= 31` value is a partial diagnostic quantity, not a certified finite-domain result.

From the recorded partial values:

```text
block_8_15_J_m2:  -1.1866684680098608e-07
block_16_31_J_m2: -2.0057575466820010e-07
block_ratio_16_31_over_8_15: 1.6902425578441627
formal_ratio_threshold: 0.8
```

Both formal integrated block ratios exceed the required threshold. The absolute Matsubara tail is unresolved.

## Matched-common-q supporting diagnostic

```text
formal_matsubara_certificate: false
common_q_count_n0_to_31: 2592
common_q_count_n16_to_31: 2592
```

```text
absolute_block_ratio_8_15_over_4_7: 1.5453556908913864
per_frequency_ratio_8_15_over_4_7: 0.7726778454456932
absolute_block_ratio_16_31_over_8_15: 1.3407764376020082
per_frequency_ratio_16_31_over_8_15: 0.6703882188010041
high_frequency_geometric_ratio: 0.969493526683795
```

These common-q diagnostics support the high-frequency decay analysis but do not replace the integrated finite-domain Matsubara closure test.

## Frozen scientific conclusion

```text
outer-Q finite-domain integration through n <= 15: available as diagnostic evidence
n <= 31 finite-domain closure: incomplete by one request / 8192 points
absolute Matsubara tail: unresolved
formal Matsubara certificate: false
old campaign continuation: forbidden
new n=32..63 extension: not planned
production_casimir_allowed: false
```

Earlier provisional labels that treated unrelated diagnostic scores as approximately `14.20` and `11.08 nJ/m^2` are invalid and must not be used as energies.

## Final retention and cleanup outcome

Retained:

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

Deleted after exact preflight revalidation:

```text
outputs/casimir/production/campaign-3a6260fd6ec8/
/home/liubx25/casimir-plans/production_plan_0deg.json
/home/liubx25/campaign-3a6260fd6ec8_spm_matched_q.csv
/home/liubx25/campaign-3a6260fd6ec8_matsubara_extract.json
12 generated TODO 1 intermediate reports
```

Cleanup identities:

```text
prune_plan_sha256: a0b8663744c3424c913b443de55883e36d43f81b817a87136c35039908a810d7
prune_execution_sha256: 9331cff9210505d02b987c797eda412ee9c2158bf9e705a16aa3d67661f1c29b
post_cleanup_audit_sha256: c6d1539fd7f28c5221940ddcb292808a1bcbe84c3d1a364ca3bf702b926ee78c
final_inventory_rows: 27
final_inventory_sha256: a5c3ff9d9986b2442e17c0c49ee7203700324c33cb0117d18a534531868f6330
final_manifest_sha256: fcc3b4b99ae4ba36c2ad5999cfd93c0cf9c50ba5201917f2676ebc4f3a818343
```

The post-cleanup audit verified that all 16 deletion targets are absent, all retained evidence still matches its approved digest, the canonical archive is intact, the final inventory matches the execution record, the tracked worktree is clean, and the production root contains only an empty `.locks/` directory.

```text
cleanup_state: completed
post_cleanup_audit_completed: true
todo1_complete: true
```

TODO 1 is closed. Any subsequent work must begin as a new formal workflow and must not import, resume, or extend this diagnostic baseline.
