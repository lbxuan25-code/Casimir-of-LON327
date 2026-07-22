# Full-Casimir runtime progress contract

This document records TODO item 6 for the formal production workflow. Progress is an
observation layer over the already-selected scientific route; it is not an acceptance
criterion and it never requests an extra microscopic point, quadrature evaluation,
certificate, or tail fit.

## Lifecycle states

Campaign snapshots keep the following case states separate:

- `queued`;
- `running`;
- `production_authorized`;
- `numerically_unresolved`;
- `diagnostic_only`;
- `engineering_failed`.

A numerically converged but formally unauthorized result is therefore never counted as
completed production work. Counts are reported both globally and separately for each
pairing.

## Nested activity

The active stack follows the real controller nesting:

```text
Matsubara block
→ outer-Q cutoff
→ joint radial/angular controller
→ radial run
→ microscopic request
→ certifier batch
```

Each layer records only values already known to that controller: block or cutoff
position, angular/radial settings, provider counters, selected-N distribution,
unresolved-reason counts, certificate paths, and error-bound-to-budget ratios.
Adaptive layers expose their current position and frozen ceiling rather than claiming a
misleading global percentage or exact case ETA.

## Persistent artifacts

Every formal campaign writes:

```text
<campaign>/progress.json
<campaign>/progress.events.jsonl
<campaign>/runs/<case>/progress.json
<campaign>/runs/<case>/progress.events.jsonl
```

`progress.json` is an atomic latest-state snapshot. `progress.events.jsonl` is an
append-only structured history. Both carry monotonic `state_sequence` values and UTC
timestamps. `last_progress_at_utc` is separate from `last_heartbeat_at_utc`, so a live
process is not confused with a process that is still making scientific progress.

The default cadence is:

- terminal refresh: at most once every 2 seconds in a TTY;
- progress snapshot: every 10 seconds or immediately at milestones;
- heartbeat: every 30 seconds;
- non-TTY summary: at most once every 60 seconds.

Expensive milestones, such as a completed certifier batch, are written immediately.
The reporter does not rewrite a full snapshot after every microscopic point.

## Read-only status command

```bash
python -m scripts.full_casimir status --campaign <campaign-id>
python -m scripts.full_casimir status --campaign <campaign-id> --watch
python -m scripts.full_casimir status --campaign <campaign-id> --json
```

`status` reads only `progress.json`. It cannot create, resume, retry, or alter formal
work and is not a competing calculation entrypoint.

## Failure isolation and ownership boundary

Core progress callbacks are fail-isolated: an observer exception cannot change a
scientific result. Campaign progress files are operational artifacts and are not part
of the scientific-policy SHA or certified-point cache identity.

The progress heartbeat remains visibility evidence only. TODO item 7 adds a separate,
immutable campaign-owner record and token-specific lock heartbeat under
`production/.locks/`. Lock ownership, stale takeover, bounded retry and atomic-cache
recovery therefore do not depend on mutable progress snapshots. TODO item 8 excludes
progress and lock files from the authoritative scientific artifact manifest while
preserving them as operational evidence.

## Contract tests

The test suite verifies:

- observer callbacks do not add scientific calls;
- nested activity and lifecycle counts survive round-trip serialization;
- SPM and d-wave are counted independently;
- selected-N and unresolved-reason summaries are persisted;
- `status` is read-only;
- reporter failures cannot alter the core result path;
- synthetic event overhead stays below one percent of a conservative short-batch
  runtime model.
