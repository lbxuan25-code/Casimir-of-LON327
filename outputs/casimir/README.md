# Casimir output layout

`outputs/casimir/` is the local generated-data root. Generated data is not committed
to Git; the repository tracks only this layout contract and the code that creates,
audits, verifies, archives, or explicitly removes data.

## Formal campaign layout

```text
outputs/casimir/
├── README.md
├── production/
│   ├── .locks/
│   │   ├── <campaign-id>.lock.json
│   │   ├── <campaign-id>.<token>.heartbeat.json
│   │   └── history/
│   └── <campaign-id>/
│       ├── campaign.json
│       ├── policy.json
│       ├── progress.json
│       ├── progress.events.jsonl
│       ├── plans/
│       │   └── <plan-sha256>.json
│       ├── runs/
│       │   └── <physical-case>/
│       │       ├── identity.json
│       │       ├── config.json
│       │       ├── manifest.json
│       │       ├── result.json
│       │       ├── summary.json
│       │       ├── progress.json
│       │       ├── progress.events.jsonl
│       │       └── cache/
│       │           ├── identity.json
│       │           ├── certified_points.json
│       │           └── certified_points.telemetry.json
│       └── reports/
│           ├── energy_cases.csv
│           ├── recovery.json
│           ├── source_proof.json
│           ├── artifact_manifest.json
│           └── reproducibility.json
├── archive/
├── catalog/
├── reports/
└── postprocessed/
```

The active lock and heartbeat exist only while one process owns the campaign. Stale
owner records are archived before explicit takeover. Progress files and lock records
are operational evidence; they are deliberately excluded from the authoritative
scientific artifact manifest.

A formal campaign is created only through:

```bash
python -m scripts.full_casimir plan ...
python -m scripts.full_casimir run --plan ... --confirm-plan-sha256 ... --fresh
```

The same campaign is continued only through the identical plan and scientific
identity using `--resume`. Formal execution never scans, imports, migrates, extends,
or reinterprets old profile caches. It resumes from the last atomically committed
certified-point cache and writes a read-only recovery report before execution.

Verify the final source and artifact proof without starting calculation:

```bash
python -m scripts.full_casimir proof --campaign <campaign-id>
```

The former `runs/<profile-case>/`, `workflow_logs/`, pilot-extension, qualification,
background-wrapper, and versioned-profile calculation routes are historical only and
are not valid targets for new work.

## Output layout tools

All layout operations are reached through the unified top-level command.

Read-only audit:

```bash
python -m scripts.full_casimir layout audit
```

Generate a migration plan for explicitly recognized historical root entries:

```bash
python -m scripts.full_casimir layout plan
```

Stage and byte-verify the planned archive copies:

```bash
python -m scripts.full_casimir layout stage \
  --plan-path outputs/casimir/catalog/output_layout_migration_plan.json \
  --confirm-plan-sha256 <LAYOUT_PLAN_SHA256>
```

Generate and execute a separately confirmed source-removal plan only after staging
verification succeeds:

```bash
python -m scripts.full_casimir layout finalize-plan \
  --migration-plan-path outputs/casimir/catalog/output_layout_migration_plan.json \
  --stage-execution-path outputs/casimir/catalog/output_layout_stage_execution.json

python -m scripts.full_casimir layout finalize \
  --plan-path outputs/casimir/catalog/output_layout_finalize_plan.json \
  --confirm-plan-sha256 <FINALIZE_PLAN_SHA256> \
  --confirm-delete REMOVE_STAGED_LEGACY_ROOT_ENTRIES
```

## Diagnostics and audits

Read-only diagnostics and convergence audits are also routed through the unified
command:

```bash
python -m scripts.full_casimir diagnose --run-dir <run-directory>
python -m scripts.full_casimir audit --run-dir <run-directory> ...
```

They must not modify authoritative cache or result artifacts and must fail closed on
cache misses rather than launching new microscopic work.

Historical three-shift evidence can be replayed under the frozen two-shift policy
before old outputs are removed:

```bash
python -m scripts.full_casimir shift-audit \
  --input <old-certified-points.json> \
  --output <two-shift-replay.json>
```

This is an evidence-only replay and never seeds a new formal campaign.
