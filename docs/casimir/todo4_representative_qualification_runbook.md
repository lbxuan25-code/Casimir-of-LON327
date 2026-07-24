# TODO 4 representative qualification runbook

## Scope

This runbook prepares and executes the narrow diagnostic qualification required
before TODO 4 can be reviewed for completion. It is not a production Casimir
scan and cannot authorize physical observables.

```text
diagnostic_only: true
valid_for_casimir_input: false
production_casimir_allowed: false
observable_error_budget_calibrated: false
```

The frozen manifest is:

```text
validation/configs/casimir/todo4_representative_v1.json
```

The only supported command surface is:

```text
python -m validation diagnostic todo4-representative-qualification <action>
```

## Frozen representative matrix

The direct qualification is sparse rather than a full Cartesian product:

```text
pairing: spm, dwave
Matsubara index: n=0, n=1
case A: q_lab=(0.02, 0.0), angles=(0.0, 0.0)
case B: q_lab=(0.015, 0.025), angles=(0.0, 0.31)
distances: 50 nm, 100 nm, 200 nm
legacy comparison distance: 100 nm
```

The fixed-outer replay uses one deliberately small grid:

```text
pairing: spm
Matsubara index: n=0, n=1
separation: 100 nm
u_max: 2
radial order: 1
angular order: 4
angular offset fraction: 0.5
angles: (0.0, 0.0)
```

The generated campaign contains:

```text
5 geometry plans
16 geometry points
20 exact response identities
10 pairing/exact-q populate groups
32 distance updates
```

Distances do not increase material-response count. The fixed-outer grid is kept
parallel because nonzero-angle correctness is already exercised by the direct
oblique case; duplicating the rotation in every outer node would add microscopic
work without adding a new contract.

## Scientific response policy

The manifest freezes:

```text
T = 40 K
Delta0 = 0.1 eV
eta = 1e-8 eV
Matsubara indices = (0, 1)
N ladder = (128, 192, 256)
required consecutive passes = 2
envelope levels = 3
ordered shifts = ((0.5,0.5), (0.25,0.75), (0.75,0.25))
canonical reduction block size = 4096
microscopic model = symmetry_bdg_2band
```

The full material-response, convergence, geometry, matrix and unit-aware
fixed-outer tolerances are explicit in the manifest. Changing any tracked
manifest field requires a new frozen plan and a new plan SHA.

## Staged workflow

### 1. Freeze plan

`plan` requires a clean tracked source tree. It writes a deterministic
`qualification_plan.json` containing source commit, manifest SHA, every exact
response identity, all point definitions and structural operation counts.
Existing different plan content is never overwritten.

### 2. Read-only preflight

The first `preflight` opens the TODO 3 store in strict read-only mode and writes:

```text
cache_preflight_before.json
cache_miss_manifest.json
```

Cache misses are expected at this stage. The command does not create cache
entries and does not call microscopic integration.

### 3. Populate exact misses

`populate` is the only stage allowed to call the TODO 2 response ladder and
write certified TODO 3 artifacts. Work is partitioned deterministically over 10
pairing/exact-q groups. Each group evaluates both requested Matsubara sectors and
persists only established response certifications.

Unresolved responses and exceptions are written to the shard report and cause a
nonzero exit. The campaign must not substitute a nearby q, change the N ladder,
relax tolerances or delete a failing representative point.

### 4. Qualification-ready preflight

The second preflight uses `--require-complete`. It requires:

```text
zero exact response misses
same working N for the two plates of every geometry point
same exact primary shift for the two plates of every geometry point
```

The compatibility result is written to:

```text
legacy_compatibility.json
```

If a rotated plate pair establishes at different working N or primary shifts,
execution stops before any legacy replay. No common-N/common-shift search is
performed silently.

### 5. Strict read-only geometry

`geometry` loads certified responses in strict read-only mode, constructs unique
reflections and prepared pairs, evaluates all distances, and performs scalar
versus batch qualification. Its structural counters must retain:

```text
microscopic_integration_call_count = 0
response_certification_call_count = 0
cache_write_count = 0
```

### 6. Matched legacy replay

`legacy` reconstructs one old-route microscopic point for each planned geometry
point using exactly the certified working N and exact primary shift. Direct cases
compare at 100 nm because the reflection matrices and round-trip product are
distance independent; scalar-versus-batch qualification already covers all
three distances. Fixed-outer points use their sole 100 nm distance.

The 16 points are independently shardable. A failed point writes an explicit
error artifact and causes that shard to exit nonzero.

### 7. Verification and fixed-outer reduction

`verify` requires every geometry and legacy point artifact, repeats read-only
cache and evidence checks, and assembles the real old/new fixed-outer arrays. It
then compares:

```text
dimensionless node logdet
outer integral in m^-2
Matsubara contribution in J/m^2
finite partial total in J/m^2
```

The final result is:

```text
verification.json
```

A pass remains diagnostic, finite-frequency and tail-free.

## Recommended local resource policy

The campaign has 10 response-population groups and 16 legacy points. On a
32-GB workstation, use `shard_count=10` and `shard_count=16` for deterministic
one-item shards, but run at most four shards concurrently. This preserves
restartability without launching ten or sixteen large microscopic jobs at once.

No `.venv` should be created. Use the existing Conda environment. Long stages
should run under `nohup` or an equivalent persistent shell and keep independent
per-shard logs.

## Generated artifacts

The output directory contains:

```text
qualification_plan.json
qualification_plan.sha256
plan_run.json
cache_preflight_before.json
cache_miss_manifest.json
populate/shard_*.json
cache_preflight_after.json
legacy_compatibility.json
geometry/*.json
geometry/*.npz
geometry_summary.json
legacy/<plan>/*.json
legacy/shard_*.json
fixed_outer/*.json
verification.json
```

The response cache is separate from the report directory. Runtime logs and cache
content remain local and ignored by Git.

## Stop conditions

Stop and inspect artifacts rather than continuing when any of the following
occurs:

- response certification is unresolved;
- hard physical validation fails;
- cache identity or checksum validation fails;
- the second preflight still reports a miss;
- a plate pair has different working N or primary shift;
- scalar-versus-batch qualification fails;
- matched legacy reflection/product/logdet qualification fails;
- fixed-outer replay fails its unit-aware policy.

The qualification set must not expand automatically to search for easier points.
