# Full outer Casimir integration handoff

## 1. Repository state

```text
repository = lbxuan25-code/Casimir-of-LON327
branch = refactor/two-band-casimir-contract
PR = #2, draft, unmerged
```

Current hard state:

```text
arbitrary_q_performance_contract = not_yet_qualified
arbitrary_q_microscopic_contract = not_yet_qualified
qualified_outer_q_envelope_established = False

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

A full outer-integration trial is **not yet authorized**. The next work is target-machine formal microscopic evidence, followed by a q-envelope contract derived from the intended outer separation/tail range.

## 2. Retained physical chain

The typed two-band path provides:

- primitive crystal blocks `K_SS`, `K_Seta`, `K_etaS`, `K_etaeta`;
- post-integral amplitude/phase Schur correction and `K_eff`;
- RHS-aware finite-q Ward validation;
- exact zero-Matsubara thermodynamic divided differences;
- zero-mode density/stiffness sheet response and static reflection;
- positive-Matsubara conductivity sheet response and reflection;
- common lab LT tangential-electric reflection basis;
- signed-real passive trace-log through `lno327.casimir.lifshitz_integrand.passive_sheet_logdet`.

Zero frequency must never use `sigma=-K/xi`.

## 3. Retained commensurate reference

The complete-orbit reference remains:

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
panel_count = 16
```

It integrates a full transverse period with no even, C4, axis, diagonal or q-direction symmetry reduction. Exact zero and positive Matsubara values share eigensystems. Child processes evaluate full q orbits and the parent performs ordered complex Kahan reduction.

The complete-orbit and arbitrary-q paths use one primitive kernel and one q-workspace implementation. The operator-enabled interface is a thin wrapper, eliminating a common copied-physics path.

## 4. Arbitrary-q microscopic implementation

`ArbitraryQPeriodicBZContract-v3` supplies:

```text
exact q_crystal = R(-theta_plate) q_lab
fixed shifted even-N, N x N full periodic BZ lattice
MaterialGridCache-v3
CrystalResponseCache-v3
runtime-sized shifted Hamiltonian/eigensystem/vertex/Kubo batches
canonical deterministic primitive reduction blocks
operator identity from existing q-workspace intermediates
exact zero + positive Matsubara shared q workspace
q_lab + angle-batch persistent POSIX-fork tasks
actual child BLAS threadpool checks
```

`runtime_chunk_size` now controls real compute width. `canonical_reduction_block_size` alone controls floating-point grouping.

## 5. Formal evidence boundary

Only `ArbitraryQFormalPolicyV2` can establish formal evidence. It freezes all parameters affecting workload, physical point or pass/fail, including:

```text
absolute and relative comparison tolerances
Ward tolerances
T, delta0, eta and separation
reference panel/order/workers/task size
runtime and canonical block policy
outer, qualification-primary and qualification-audit workload identities
```

Formal runs require a clean source tree. Evidence records:

```text
git_head
git_tree_sha
tracked_index_fingerprint
source_tree_fingerprint
worktree_clean = True
```

The public gate requires identical provenance across performance, current checkout, numerical core output and post-run checkout.

The numerical core itself can emit only:

```text
diagnostic_result_passed
diagnostic_result_failed
```

Only the public clean-source gate may promote a passed result to:

```text
qualified_for_diagnostic_outer_integration
```

This promotion still leaves:

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

## 6. Required target-machine order

Set BLAS/OpenMP variables to one before Python starts, then:

1. Run the retained complete-orbit timing evidence on the current clean head.
2. Run `matsubara arbitrary-q-performance-preflight` on the same clean head.
3. Confirm the worktree remains clean and do not modify source.
4. Run the public `matsubara arbitrary-q-periodic-bz-qualification` gate with the performance manifest.
5. Inspect the formal manifest and discrete q-coverage record.

Exact commands are maintained in `scripts/casimir/README.md`.

## 7. What formal microscopic qualification checks

For each pairing it evaluates:

```text
primary N=256
primary N=384
primary N=512
audit A=(1/4,3/4) at N=512
audit B=(3/4,1/4) at N=512
```

Each complete-orbit reference, primary N, audit shift and paired result must independently pass the applicable operator, integrated Ward, strict-static, sheet, reflection and passive-logdet gates.

### Paired shifts

Each plate is paired at the linear primitive level:

```text
paired_packed = 0.5 * (packed_A + packed_B)
```

The API verifies identical material state, compatible grid identity and inversion-related formal shifts. `PairedShiftProfile-v1` sums both source evaluations while retaining one effective counterterm.

### Final two-plate observable

Qualification directly gates the nonlinear quantity consumed by future outer integration:

```text
plate 1 theta = 0 degrees
plate 2 theta = 17 degrees
common lab LT basis
logdet(I - R1 R2 exp(-2 kappa d))
```

For every Matsubara index it requires:

```text
N=256,384,512 two-plate values
N refinement
audit A and B two-plate values
primitive-paired plate 1 and plate 2
paired two-plate value
A/B sensitivity
primary N512/paired sensitivity
all source and paired plate physical gates
```

Single-plate convergence cannot substitute for this final-observable gate.

## 8. Momentum support is not an outer envelope

The implementation accepts only:

```text
|q_x| <= pi
|q_y| <= pi
```

without wrapping. This is a syntactic principal-domain support boundary, not a numerically qualified outer domain.

The current matrix covers discrete axis, generic, near-diagonal, exact-diagonal and 17-degree-rotated vectors. Its manifest explicitly keeps:

```text
qualified_outer_q_envelope_established = False
continuous_angle_coverage_established = False
outer_tail_requirement_bound = False
```

Before outer implementation, define the intended:

```text
separation range
angle range
q quadrature family
q cutoff rule
tail tolerance
```

Then derive the required `q_max` and qualify multiple radii/directions through that range. The resulting envelope manifest must contain maximum norm/component and angle coverage. A future outer builder must reject nodes outside it.

## 9. Exact-diagonal d-wave status

The retained complete-orbit evidence found response-level cut sensitivity on exact `qx=qy` d-wave directions, while neighboring off-diagonal directions were strict and tested reflection/logdet sensitivity stayed below `1e-3`.

This remains a sensitivity issue rather than permission to bypass physical gates. Exact-diagonal primitive response may be reported unresolved only when operator, integrated Ward, strict-static, reflection, logdet and shift-sensitivity gates pass.

No full outer trial should be started until the q-envelope design specifies how diagonal directions and neighboring-angle sensitivity are sampled.

## 10. Future outer architecture

After the clean-source microscopic manifest and q-envelope manifest exist, the outer layer should remain thin:

1. `OuterIntegrationConfig`: temperature, separations, angles, q quadrature/cutoff/tail and Matsubara tolerances.
2. `MicroscopicResponseProvider`: exact q/angle requests into the typed arbitrary-q backend.
3. `QQuadrature`: lab-frame q nodes/weights and envelope enforcement.
4. `MatsubaraQuadrature`: prime weight, cutoff and tail diagnostics.
5. `OuterAccumulator`: deterministic compensated summation.
6. `SensitivityVariants`: diagonal/node/outer-grid alternatives.
7. `ReportWriter`: atomic manifests and compact summaries.

The outer variable is `q_lab`. Each plate receives `q_crystal = R(-theta_plate) q_lab`; both reflections must return to the same lab LT basis before multiplication.

## 11. Required outer convergence after authorization

A diagnostic energy result will still require:

```text
all microscopic point pipelines passed
all signed-real logdet constructions passed
q radial/angular convergence
q cutoff and omitted-tail convergence
Matsubara cutoff/tail convergence
angle-grid convergence
energy convergence before torque differentiation
diagonal/shift sensitivity
serial/process deterministic equality
```

Torque needs its own absolute floor near symmetry angles.

Only the complete microscopic + q-envelope + outer convergence chain may eventually support changing:

```text
production_reference_established = True
valid_for_casimir_input = True
```

No local threshold or single manifest can change those flags by itself.

## 12. Forbidden shortcuts

Do not:

- divide by Matsubara frequency at `n=0`;
- use bare or phase-only kernels;
- substitute local-q0 response at finite q;
- silently wrap or nearest-grid-round q;
- average nonlinear reflections/logdets as a quadrature reference;
- mix plate reflection bases;
- claim `[-pi,pi]^2` is numerically qualified from the current discrete matrix;
- run formal commands on a dirty tree;
- reuse a performance manifest after any source change;
- open idle eight-worker pools for one-task audit contexts;
- restore the retired monolithic Casimir script.

## 13. Output policy

Before a production reference exists, retain only exact commands, configs, source/hardware fingerprints and compact summary/status artifacts. Raw grids, caches, full tables and figures remain local and ignored.
