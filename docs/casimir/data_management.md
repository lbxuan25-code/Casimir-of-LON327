# Casimir local data management

All generated Casimir artifacts remain local under `outputs/casimir/`. Git tracks code and layout documentation, not run data.

The data-management workflow separates two independent concepts:

- **scientific state** is derived from run artifacts: `certified`, `diagnostic_complete`, `unresolved`, `incomplete`, `corrupt`, or `unknown`;
- **lifecycle state** is an explicit human decision in a local registry: `active`, `frozen_evidence`, `superseded`, `legacy_exploratory`, `abandoned`, `scratch`, or `archived`.

An unresolved run can still be valuable frozen evidence. Conversely, a complete diagnostic may be superseded. The catalog never infers lifecycle from a directory name.

## 1. Build a read-only catalog

```bash
python -m scripts.full_casimir.data catalog \
  --write-registry-template
```

This writes:

```text
outputs/casimir/catalog/run_catalog.json
outputs/casimir/catalog/run_catalog.tsv
outputs/casimir/catalog/registry.template.json
```

The catalog records run sizes, required artifacts, schemas, hashes, physical identity, numerical policy, result state, cache counts, source commit, and non-run diagnostics/reports/logs. It does not modify any run.

## 2. Curate the local registry

Copy the template and edit it:

```bash
cp outputs/casimir/catalog/registry.template.json \
   outputs/casimir/catalog/registry.json
```

Each run receives:

```json
{
  "lifecycle_state": "active",
  "retention_action": "keep_hot",
  "note": "current 0-degree convergence evidence"
}
```

Allowed retention actions are:

- `keep_hot`: retain unpacked for active reuse;
- `keep_cold`: retain unpacked but not active;
- `archive`: create a verified compressed copy;
- `review`: no action until classified.

No automatic rule is allowed to delete a run.

## 3. Build an archive plan

```bash
python -m scripts.full_casimir.data plan
```

This refreshes the catalog and writes:

```text
outputs/casimir/catalog/archive_plan.json
```

The plan includes the exact source-tree digest, file count, byte count, archive destination and a `plan_sha256`. Planning is read-only.

## 4. Create verified archives

Copy the printed plan hash exactly:

```bash
python -m scripts.full_casimir.data archive \
  --plan-path outputs/casimir/catalog/archive_plan.json \
  --confirm-plan-sha256 <PLAN_SHA256>
```

A subset can be selected with repeated `--run <name>` arguments.

Archives are written under:

```text
outputs/casimir/archive/runs/<run>.tar.gz
```

Each archive has a sidecar manifest containing every source path, size, mode and SHA-256. The tool verifies the source tree has not changed since planning and verifies the archive member list after writing.

The archive command never deletes or moves the source directory.

## 5. Restore-verify before any source removal

Archive member-list verification is not sufficient for source pruning. Restore each archive into temporary storage and compare the restored tree to the per-file manifest:

```bash
python -m scripts.full_casimir.data verify \
  --plan-path outputs/casimir/catalog/archive_plan.json
```

This writes:

```text
outputs/casimir/catalog/archive_verification.json
```

The verifier rejects absolute paths, `..`, links and device entries, restores regular files and directories only, and compares every restored path, size, mode and SHA-256. Temporary restored copies are removed after verification. Original archives and source directories remain present.

## 6. Build an explicit prune plan

Source removal is a separate contract. Every run must be named explicitly:

```bash
python -m scripts.full_casimir.data prune-plan \
  --registry outputs/casimir/catalog/registry.json \
  --verification-report outputs/casimir/catalog/archive_verification.json \
  --run <RUN_NAME> \
  --run <RUN_NAME>
```

The command refuses:

- unverified archives;
- source trees changed after verification;
- archives changed after verification;
- runs not registered with `retention_action=archive`;
- `active` and `frozen_evidence` lifecycle states;
- an empty implicit selection.

It writes `outputs/casimir/catalog/prune_plan.json` and prints a new exact plan SHA-256. No source is modified during planning.

## 7. Execute an approved prune plan

Source deletion requires both the exact plan hash and a literal confirmation phrase:

```bash
python -m scripts.full_casimir.data prune \
  --plan-path outputs/casimir/catalog/prune_plan.json \
  --confirm-plan-sha256 <PRUNE_PLAN_SHA256> \
  --confirm-delete DELETE_VERIFIED_ARCHIVED_RUN_SOURCES
```

Before deleting anything, the command preflights every selected source tree, archive SHA-256 and manifest relation. It then removes only the source run directories in the exact plan. Verified archive files and manifests remain present.

The first recommended pruning tier is limited to `superseded`, `legacy_exploratory` and `abandoned` runs. Keep `frozen_evidence` unpacked until the replacement certified baseline has been accepted.

## 8. Pack oversized JSON reports

Large audit reports can retain scalar decisions in a compact JSON while moving large list-valued evidence into deterministic gzip sidecars:

```bash
python -m scripts.full_casimir.data pack-report \
  --report-path outputs/casimir/reports/convergence_audit.json \
  --threshold-mib 1
```

Default outputs are:

```text
outputs/casimir/reports/convergence_audit.compact.json
outputs/casimir/reports/convergence_audit.pack/
outputs/casimir/reports/convergence_audit.pack_manifest.json
```

Each reference records its JSON pointer, item count, uncompressed size, compressed size, value digest and sidecar SHA-256. The command reconstructs the complete object and compares its canonical digest before declaring success. The original report is preserved; removing it requires a later explicit artifact-retention decision.

## Safety contract

- run directories are treated as immutable evidence until an exact prune plan is executed;
- catalog generation and archive/prune planning are read-only;
- lifecycle decisions must be explicit in the registry;
- archive execution requires the exact archive-plan SHA-256;
- source pruning requires a successful temporary restore verification;
- source pruning requires explicit run names, an exact prune-plan SHA-256 and a literal deletion phrase;
- `active` and `frozen_evidence` runs cannot be pruned by this tool;
- a source or archive change after verification aborts pruning;
- archive creation is atomic;
- archive and report sidecars carry SHA-256 manifests;
- report packing never overwrites or removes the source report.
