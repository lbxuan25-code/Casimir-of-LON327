# Casimir local data management

All generated Casimir artifacts remain local under `outputs/casimir/`. Git tracks code and layout documentation, not run data.

The data-management workflow deliberately separates two independent concepts:

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

Each archive has a sidecar manifest containing every source path, size and SHA-256. The tool verifies the source tree has not changed since planning and verifies the archive member list after writing.

**The archive command never deletes or moves the source directory.** Source pruning is intentionally outside this first data-management contract and must only be added after archived results have been reviewed and restored successfully.

## Safety contract

- run directories are treated as immutable evidence;
- catalog generation is read-only;
- lifecycle decisions must be explicit in the registry;
- archive execution requires the exact plan SHA-256;
- a source change after planning aborts execution;
- archive creation is atomic;
- archive member verification and a SHA-256 sidecar are mandatory;
- no command in this module removes source data.
