# TODO 1 — verified zero-degree baseline findings

This note records the evidence extracted from the stopped formal SPM campaign and its read-only diagnostic replays. It does not alter the historical campaign and does not authorize production use or resume.

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

Formal campaign source:

```text
outputs/casimir/production/campaign-3a6260fd6ec8/
```

Verified source snapshot:

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

The loose external `production_plan_0deg.json` is byte-identical to the formal plan.

## Formal interruption state

The historical run manifest still says `running`, but there is no active campaign lock and no final `result.json`, `summary.json`, `source_proof.json`, `artifact_manifest.json`, or `reproducibility.json`. The evidence therefore describes an interrupted run with an atomically retained cache, not a finalized formal result. The original manifest must remain unchanged.

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

The unweighted common-q block diagnostics are:

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

## Retention classification

```text
formal campaign directory:
  canonical source candidate; archive and restore-verify before source removal

campaign-3a6260fd6ec8_spm_cache_only_joint_u6.json:
  baseline_evidence; retain

campaign-3a6260fd6ec8_spm_matched_q.json:
  baseline_evidence; retain

campaign-3a6260fd6ec8_spm_microscopic_decay.csv:
  baseline_evidence; retain as a compact cache-derived table

campaign-3a6260fd6ec8_spm_matched_q.csv:
  derived_duplicate; exactly reproduced by the JSON table

/home/liubx25/casimir-plans/production_plan_0deg.json:
  derived_duplicate; byte-identical to the formal plan

campaign-3a6260fd6ec8_matsubara_extract.json:
  transient_working candidate; retain until the final archive manifest binds its hash

/home/liubx25/casimir-output-archive-20260721-223949/:
  supporting earlier pilot/qualification history; not the stopped formal campaign
```

No source deletion is authorized by this note. Archive creation, restore verification, and an exact prune plan remain required.
