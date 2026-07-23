# TODO 3: persistent certified material-response cache

## Status

Implementation branch: `feat/todo3-persistent-response-cache`

This document freezes the persistent response-cache boundary built on the TODO 2
material/geometry split. It does not authorize a production Casimir calculation.

```text
cache_schema: material-response-cache-v1
response_status: response_certified_diagnostic
valid_for_casimir_input: false
production_casimir_allowed: false
observable_error_budget_calibrated: false
```

## Purpose

A response that has already passed response-space N/shift certification should
survive process exit and be reusable by later geometry calculations. Reusing a
response must never require rebuilding microscopic kernels, Ward objects, or BZ
workspaces, and a cache miss in strict read-only mode must never start a hidden
microscopic fallback.

## Dependency direction

```text
material_response_engine.py
        |
        v
material_response_cached_engine.py
        |
        +--> material_response_cache_identity.py
        +--> material_response_cache_store.py
                    |
                    v
          MaterialResponseSnapshot
                    |
                    v
material_geometry.py -> material_two_plate.py
```

The core TODO 2 response engine remains cache-agnostic. The cached engine is a
wrapper that performs exact lookup, sends only missing Matsubara frequencies to
the core engine, and persists only successful response-level certifications.
Geometry accepts a live `MaterialResponseSample` or a persisted
`MaterialResponseSnapshot`, but it does not import the cache writer or the
microscopic engine.

## Cache identity

`MaterialResponseCacheIdentity` is exact and content-addressed. It contains:

- pairing name;
- finite temperature encoded by exact float hexadecimal form;
- Matsubara index, exact `xi_eV`, and frequency sector;
- exact two-component `q_crystal`;
- microscopic model adapter name and material-state fingerprint;
- response-policy fingerprint;
- primitive-contract, phase-Hessian-policy, and basis identity;
- response-convergence policy and certification algorithm parameters.

The following are intentionally not representable in the identity:

- separation or distance;
- plate angles;
- `q_lab`;
- outer quadrature orders or node identifiers;
- worker count or runtime chunk size;
- filesystem path, PID, hostname, timestamps, or wall time.

A one-bit float change in `q_crystal`, temperature, or frequency changes the
identity. There is no rounding, nearest-node lookup, wrapping, interpolation, or
surrogate fallback.

## Persisted artifact

A cache entry stores a `CachedCertifiedMaterialResponse` containing:

- the canonical cache identity;
- one immutable geometry-facing `MaterialResponseSnapshot`;
- zero-frequency susceptibility/stiffness or positive-frequency conductivity
  tensors with unit-stage tags;
- sheet-response validation and hard-physical audit summary;
- working/audit N values, establishment mode, certification evidence, and audit
  provenance;
- explicit diagnostic-only safety flags.

The snapshot deliberately excludes microscopic kernels, primitive accumulators,
eigensystems, and live Ward objects. It does not fabricate those objects during
load.

## File format

Each artifact is one content-addressed NPZ file:

```text
<root>/material-response-cache-v1/ab/cd/<identity-sha256>.npz
```

The archive contains:

- UTF-8 canonical JSON manifest;
- sector-specific NumPy arrays;
- per-array dtype, shape, and SHA-256 checksums;
- manifest-payload SHA-256;
- identity payload and identity SHA-256.

Loading always uses `np.load(..., allow_pickle=False)`. The loader verifies the
schema, manifest checksum, filename SHA, requested identity, array names, dtypes,
shapes, and array checksums before constructing readonly response objects.
Unknown schema, corruption, or identity mismatch fails closed.

## Store modes

### `disabled`

No persistent lookup or write. The wrapper may run the microscopic engine and
returns certified live snapshots without persistence.

### `populate`

Read-through mode. Exact hits are loaded. Only missing Matsubara frequencies are
sent to the microscopic engine. Successfully certified misses are written
atomically; unresolved frequencies are not written to the certified library.

### `read_only`

Strict cache-only mode. A miss raises `MaterialResponseCacheMiss`. The loader
does not create directories, modify files, repair artifacts, or start a
microscopic fallback.

## Atomic write and concurrency contract

A new artifact is written to a unique temporary file in the final directory,
flushed, fsynced, and fully reloaded for validation. The writer then acquires an
exclusive per-identity lock. If no final artifact exists, the temporary file is
installed with same-directory `os.replace`, followed by a parent-directory
fsync.

If an artifact already exists, it is loaded and compared under the recorded
response-convergence policy. A compatible artifact is reused. An incompatible
response raises `MaterialResponseCacheConflictError`; existing data is never
overwritten. Existing lock files are not silently deleted or treated as stale.

Failures before the atomic replace cannot create a loader-visible final file.

## Typed failures

The public failure surface distinguishes:

- `MaterialResponseCacheMiss`;
- `MaterialResponseCacheReadOnlyError`;
- `MaterialResponseCacheIdentityError`;
- `MaterialResponseCacheCorruptionError`;
- `MaterialResponseCacheConflictError`;
- `UnsupportedMaterialResponseCacheSchema`;
- `MaterialResponseCacheLockError`.

These conditions are not converted into silent misses.

## Completion gate

TODO 3 may be marked complete only when all of the following hold:

- cache identity is exact, geometry-free, schema-versioned, and deterministic;
- zero- and positive-Matsubara artifacts round-trip with readonly arrays and
  preserved response values, validation, and unit-stage tags;
- atomic write, lock, strict read-only, corruption, identity-mismatch, and
  conflict behavior are tested fail closed;
- a cold run persists certified responses and a warm run succeeds with the
  microscopic engine forbidden;
- partial hits send only missing frequencies to microscopic evaluation;
- unresolved responses never enter the certified library;
- loaded responses reproduce live reflection and two-plate logdet without a
  microscopic call;
- architecture tests preserve one-way dependencies and forbid geometry fields in
  cache identity;
- repository-wide tests and GitHub Actions pass;
- documentation and current-route status are updated;
- production authorization remains false.

## Explicit exclusions

TODO 3 does not implement:

- frequency interpolation, Chebyshev, IR, or DLR compression;
- nearest-q reuse, q interpolation, or surrogate responses;
- multi-angle or multi-distance orchestration beyond direct replay;
- migration or promotion of old point caches or `CrystalResponseCache`;
- persistence of unresolved intermediate ladders in the certified library;
- observable-level error allocation or production admission;
- true zero-temperature microscopic calculations.
