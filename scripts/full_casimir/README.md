# Unified Casimir command

The only authorized user-facing Casimir calculation command is:

```bash
python -m scripts.full_casimir
```

No package console command, direct single-case command, pilot workflow, qualification
workflow, background wrapper, cache migration helper, or versioned profile runner is a
supported calculation path.

## Formal calculation sequence

Create and review an immutable plan:

```bash
python -m scripts.full_casimir plan \
  --pairings spm dwave \
  --distances-nm 20 \
  --angles-deg 0 \
  --plan-output production_plan.json
```

The command prints the exact `plan_sha256`. Start a new campaign only from an empty
formal campaign directory:

```bash
python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 <PLAN_SHA256> \
  --fresh
```

Resume the same scientific campaign with:

```bash
python -m scripts.full_casimir run \
  --plan production_plan.json \
  --confirm-plan-sha256 <PLAN_SHA256> \
  --resume
```

`plan` and `run` are the only commands that define or execute formal Casimir work.
The top-level command also exposes read-only monitoring, diagnostics, audits,
data/layout tools, and post-processing of existing results.

## Runtime progress

A formal run writes campaign- and case-level `progress.json` snapshots plus append-only
`progress.events.jsonl` event streams. The terminal shows the current nested activity,
microscopic/cache counters, unresolved-reason counts, and normalized error-budget
ratios without performing extra scientific work.

Read one persisted snapshot:

```bash
python -m scripts.full_casimir status \
  --campaign <campaign-id>
```

Continuously monitor it or consume the machine-readable form:

```bash
python -m scripts.full_casimir status --campaign <campaign-id> --watch
python -m scripts.full_casimir status --campaign <campaign-id> --json
```

`status` is read-only. Heartbeats report liveness only; locking, retry ownership, and
crash recovery remain separate restart-policy concerns.

## Historical evidence

Historical outputs are never imported into a formal campaign. Before old three-shift
outputs are archived or removed, their stored histories may be replayed without new
microscopic calculation:

```bash
python -m scripts.full_casimir shift-audit \
  --input <old-certified-points.json> \
  --output <two-shift-replay.json>
```

## Internal modules

Modules under `lno327.casimir` and the command-handler modules under this directory
are implementation details. They may be imported by tests and the unified dispatcher,
but they are not independent operational entrypoints.
