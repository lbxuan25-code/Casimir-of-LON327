# Arbitrary-q periodic BZ implementation decision

Status remains:

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

## Frozen architecture

- Exact `q_crystal = R(-theta) q_lab`; no q/angle rounding, wrapping, nearest-commensurate substitution, or primitive interpolation.
- Fixed shifted `N x N` periodic midpoint BZ lattice; even `N` only.
- Primary shift `(1/2,1/2)` with explicit adjacent `k/-k` ordering.
- Audit shifts `(1/4,3/4)` and `(3/4,1/4)` are evaluated independently. Their primary paired estimate is formed as `0.5 * (packed_A + packed_B)` at the linear primitive level, followed by exactly one phase-Hessian/Schur/sheet/reflection/logdet pipeline. Nonlinear reflection or logdet averages are diagnostic spreads only and are never called a quadrature reference.
- One shared quadrature-independent primitive kernel for the retained complete-orbit path and the periodic-BZ path.
- Full linear primitive accumulation precedes phase-Hessian, Schur, sheet, reflection, and logdet processing.
- One readonly q-independent material cache per material/pairing/T/grid/config fingerprint.
- Q-dependent arrays are generated in streamed canonical reduction blocks and released.
- Goldstone/HS counterterm is added exactly once per full BZ result.
- Exact zero and all positive Matsubara frequencies share each shifted q workspace.
- Persistent POSIX-fork process parallelism uses `q_lab + angle_batch` tasks; BLAS/OpenMP threads are fixed to one.
- When a material cache is supplied, the integration entry point uses `material_cache.grid` and performs no additional `O(N^2)` grid construction.

## Accepted amendments and review corrections

1. `pointwise Ward` means the exact normal-state Peierls operator identity only. Integrated response Ward closure remains a separate convergence and physical gate.
2. The production parallel task is centered on one `q_lab` and an angle batch, so the fixed plate response is evaluated once per task.
3. Runtime memory chunks do not define floating-point reduction. A fixed canonical reduction block controls numerical grouping; runtime chunks only group one or more canonical blocks.
4. The Peierls operator identity is computed from Hamiltonians and vertices already owned by the q workspace; it does not trigger a duplicate Hamiltonian/vertex pass.
5. Every complete-orbit reference, every primary `N`, both audit shifts, and the paired-primitive result must independently pass operator, integrated Ward, strict-static, sheet, reflection and passive-logdet gates.

## Frozen formal policy

Only `ArbitraryQFormalPolicyV1` may authorize a formal manifest. CLI values may be stricter, but values looser than the frozen limits cannot establish a formal pass.

Performance policy includes:

```text
pairings: spm,dwave
N >= 128
q tasks >= 8
workers >= 4
Matsubara includes 0,1,2,4,8
canonical reduction block = 4096
runtime chunks include 4096 and 16384
minimum speedup >= 4
minimum CPU/wall >= 4
maximum pool overhead <= 0.05
```

Numerical policy includes:

```text
pairings: spm,dwave
N values include 256,384,512
reference nk = 1256
reference order >= 384
Matsubara includes 0,1,8
primitive rtol <= 1e-3
reflection rtol <= 3e-4
logdet rtol <= 3e-4
diagonal observable rtol <= 1e-3
```

The formal performance manifest records the policy id, config fingerprint, exact command, hardware fingerprint, git head, execution strategy, worker/thread policy and actual BLAS threadpool report. The numerical gate requires a compatible same-head manifest.

The numerical core itself writes only:

```text
diagnostic_result_passed
diagnostic_result_failed
```

Only the public same-head formal gate may promote a passed result to:

```text
qualified_for_diagnostic_outer_integration
```

## Cache identity

`MaterialGridCache-v2` includes:

```text
spec class and complete public/explicit numerical state
all two-band model parameters
hopping/basis/convention payload when provided
ansatz name, phase vertex and form-factor state
pairing amplitudes
thermodynamic config and options
grid fingerprint, N, shift and ordering
```

A generic `metadata()` value is not treated as a complete numerical identity.

`CrystalResponseCache-v2` includes:

```text
material fingerprint
exact q bytes
exact Matsubara bytes
phase-Hessian policy
canonical reduction block
operator-Ward atol/rtol
primitive contract version
```

Runtime chunk size is deliberately excluded because it must not change the numerical definition. Signed zero is normalized; NaN/Inf are rejected; no decimal rounding is allowed.

## Performance evidence

The formal performance preflight records and gates:

```text
actual q-workspace eigensystem-call counters
short versus full Matsubara-batch eigensystem equality
actual BLAS runtime threadpool counts
worker RSS/PSS
serialized IPC payload bytes
parent collection overhead
cache-on versus cache-off numerical equality and timing
serial/process and runtime-chunk equality
persistent-pool speedup and CPU/wall ratio
```

After the operator-audit integration change, both the retained complete-orbit timing preflight and the arbitrary-q performance preflight must be rerun on the target WSL/Linux machine.

## Validated microscopic momentum domain

The microscopic backend currently rejects any component outside:

```text
|q_x| <= pi
|q_y| <= pi
```

It never silently wraps q. Before an outer production calculation is allowed, its `Q` cutoff and tail convergence must demonstrate that every rotated `q_crystal` remains inside the validated microscopic domain and that the omitted tail is negligible before a BZ boundary is reached. Umklapp/local-field extensions are outside this contract.

## Qualification order

1. Shared-kernel complete-orbit regression and renewed real-hardware timing.
2. Tiny one-shot/streamed, cache, key, q-domain, spm/dwave Ward and two-plate physical tests.
3. Persistent-pool formal performance preflight.
4. Same-head formal gate verification.
5. Commensurate regression against complete-orbit.
6. Arbitrary-q refinement `N=256,384,512` and paired-primitive shift audit.
7. Two-plate `q_lab`, `theta=(0,17 deg)` common-lab-basis physical pipeline.
8. Only after all gates pass may the backend be marked `qualified_for_diagnostic_outer_integration`.
