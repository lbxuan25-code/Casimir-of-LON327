# Arbitrary-q periodic BZ implementation decision

Status remains:

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

## Frozen architecture

- Exact `q_crystal = R(-theta) q_lab`; no q/angle rounding or primitive interpolation.
- Fixed shifted `N x N` periodic midpoint BZ lattice; even `N` only in v1.
- Primary shift `(1/2,1/2)` with explicit adjacent `k/-k` ordering.
- Audit shifts `(1/4,3/4)` and `(3/4,1/4)` evaluated independently and averaged only at the observable audit layer.
- One shared quadrature-independent primitive kernel for the qualified complete-orbit path and the new periodic-BZ path.
- Full linear primitive accumulation precedes phase-Hessian, Schur, sheet, reflection, and logdet processing.
- One readonly q-independent material cache per material/pairing/T/grid/config fingerprint.
- Q-dependent arrays are generated in streamed canonical reduction blocks and released.
- Goldstone/HS counterterm is added exactly once per full BZ result.
- Exact zero and all positive Matsubara frequencies share each shifted q workspace.
- Persistent POSIX-fork process parallelism uses `q_lab + angle_batch` tasks; BLAS/OpenMP threads are fixed to one.

## Three accepted amendments

1. `pointwise Ward` means the exact normal-state Peierls operator identity only. Integrated response Ward closure remains a separate convergence gate.
2. The production parallel task is centered on one `q_lab` and an angle batch, so the fixed plate response is evaluated once per task.
3. Runtime memory chunks do not define floating-point reduction. A fixed canonical reduction block controls numerical grouping; runtime chunks only group one or more canonical blocks.

## Cache keys

Material cache keys include material/model parameters, pairing amplitudes and phase-vertex convention, temperature/config/options, `N`, shift, ordering and BZ convention.

Crystal-response keys use canonicalized IEEE-754 float64 bytes for exact q and frequency values. Signed zero is normalized; NaN/Inf are rejected; no decimal rounding is allowed.

## Qualification order

1. Shared-kernel complete-orbit regression.
2. Tiny grid/cache/chunk/operator-Ward correctness.
3. Persistent-pool deterministic and performance preflight.
4. Commensurate regression against complete-orbit.
5. Arbitrary-q refinement `N=256,384,512` and paired-shift audit.
6. Two-plate `q_lab`, `theta=(0,17 deg)` common-lab-basis physical pipeline.
7. Only after all gates pass may the backend be marked `qualified_for_diagnostic_outer_integration`.
