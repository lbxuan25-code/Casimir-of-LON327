# Reconstructable audit-report retention

Oversized JSON audit reports may be externalized with:

```bash
python -m scripts.full_casimir.data pack-report \
  --report-path outputs/casimir/reports/convergence_audit.json \
  --threshold-mib 1
```

The pack manifest records the original report path, byte count, file SHA-256, canonical JSON-value digest, compact-report SHA-256 and every compressed sidecar. Packing reconstructs the full JSON object before declaring success and does not remove the original source.

## Plan source-report removal

After reviewing the compact report and pack manifest, create a separate prune plan:

```bash
python -m scripts.full_casimir.data report-prune-plan \
  --manifest-path \
    outputs/casimir/reports/convergence_audit.pack_manifest.json
```

The command re-verifies:

- the pack-manifest self digest;
- compact-report path, byte count and SHA-256;
- every sidecar path, byte count and SHA-256;
- full JSON reconstruction and canonical value digest;
- original source path, byte count, SHA-256 and JSON value digest.

It writes a plan next to the manifest and prints a new `plan_sha256`. The original report remains present.

## Execute an approved report prune

```bash
python -m scripts.full_casimir.data report-prune \
  --plan-path \
    outputs/casimir/reports/convergence_audit.pack_manifest.prune_plan.json \
  --confirm-plan-sha256 <REPORT_PRUNE_PLAN_SHA256> \
  --confirm-delete DELETE_RECONSTRUCTABLE_PACKED_REPORT_SOURCE
```

Before unlinking the original report, execution repeats reconstruction and all source, compact and sidecar checks. It removes only the exact source report named by the plan. The compact report, gzip sidecars, pack manifest and execution record remain present.

This contract is intended for derived reports whose complete JSON value is reproducible from the compact representation. It does not apply to run caches, raw microscopic histories or any file lacking a verified report-pack manifest.
