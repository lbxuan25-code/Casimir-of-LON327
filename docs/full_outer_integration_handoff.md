# Full outer Casimir integration handoff

## 1. Repository state

Repository: `lbxuan25-code/Casimir-of-LON327`

Branch: `refactor/two-band-casimir-contract`

Draft PR: `#2`

Use the current branch head as the source of truth. Any preflight manifest generated on an older head is invalid and must be regenerated before a formal scan.

Hard state at handoff:

```text
diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```

A diagnostic full outer-integration trial is allowed. No production or final-Casimir claim is allowed yet.

## 2. What is ready

The retained two-band finite-q path provides:

- typed primitive crystal blocks `K_SS`, `K_Seta`, `K_etaS`, `K_etaeta` and Schur-corrected `K_eff`;
- RHS-aware finite-q Ward validation on the same microscopic quadrature;
- exact zero-Matsubara divided differences;
- static density/stiffness sheet response and static reflection;
- positive-Matsubara conductivity sheet response and reflection;
- common lab LT tangential-electric reflection basis;
- signed-real passive trace-log through `lno327.casimir.lifshitz_integrand.passive_sheet_logdet`;
- one total-Matsubara complete-orbit callback sharing eigensystems across `n=0` and positive frequencies;
- deterministic POSIX-fork transverse-node execution with ordered parent Kahan reduction.

Zero frequency must never use `sigma=-K/xi`.

## 3. Qualified validation entry points

Only these public commands belong to the pre-outer main path:

```bash
python -m validation matsubara total-orbit-timing-profile --help
python -m validation matsubara matsubara-orbit-gauss-crosscheck --help
python -m validation matsubara orbit-gauss-preflight --help
python -m validation matsubara total-orbit-gauss-scan --help
```

Ward and independent static checks remain available under `ward` and `static`.

Historical positive-only aliases are no longer public. D-wave adaptive, integrand-profile and angular-width tools live under `python -m validation diagnostic ...`; they are forensic tools and must not become runtime dependencies of the outer integrator.

## 4. Retained microscopic quadrature contract

The qualified main path uses a full-period equal-panel composite Gauss-Legendre transverse integral.

```text
screen: C64 / C96
medium: C160 / C192
hard:   C320 / C384
panel_count = 16
```

No even, C4, axis, diagonal or q-direction symmetry reduction is allowed inside this microscopic integral. A child evaluates one full commensurate q orbit; the parent reduces original node order with complex Kahan summation.

For process parallelism set all BLAS/OpenMP thread counts to one.

## 5. Last formal total-Matsubara evidence

The last complete scan used:

```text
pairings = spm, dwave
nk = 1256
Matsubara indices = 0,1,2,4,8,16,32
Gauss stages = 64/96, 160/192, 320/384
q cases = (1,0), (1,1), (2,1), (3,2), (6,4), (6,0), (6,6), (9,6)
```

Result:

```text
all closure checks passed = True
all observable checks passed = True
spm strict cases = 8/8
dwave strict cases = 6/8
```

The only response-level unresolved cases were exact d-wave diagonal directions `(1,1)` and `(6,6)`. Their Ward, exact-static Ward, sheet, reflection and logdet pipelines passed.

Do not reuse the old preflight manifest after this handoff commit. Regenerate it on the current head.

## 6. Exact-diagonal d-wave finding

A two-cut C384 angular-width screen evaluated `n=0,1` at:

```text
(6,6), (12,12), (24,24),
(25,24),
(13,12), (12,13),
(13,11), (11,13),
(14,10), (10,14)
```

Exact diagonal results:

```text
(6,6):   static 1.482e-2, positive 4.434e-3, R 8.743e-5, logdet 1.738e-4
(12,12): static 2.102e-2, positive 1.551e-2, R 1.586e-4, logdet 3.159e-4
(24,24): static 4.249e-2, positive 4.727e-2, R 2.551e-4, logdet 5.092e-4
```

Nearest tested off-diagonal direction:

```text
(25,24)
angular offset from 45 degrees = 1.1691393279 degrees
static cut drift = 1.528e-6
positive cut drift = 1.841e-6
R cut drift = 5.804e-6
logdet cut drift = 9.364e-9
classification = response_strict
```

Mirror directions agree. All observables in the screen are stable below `1e-3`.

Interpretation:

- the integrand is periodic to machine precision;
- panel-boundary left/right probes are continuous at approximately `1e-6` for a `2e-7` probe separation;
- the response instability is consistent with an exact-diagonal commensurate/nodal sampling anomaly rather than a finite-width angular wedge at tested resolution;
- the local response reference remains unresolved, but observable sensitivity is strongly suppressed.

This issue must not block the first diagnostic outer integral.

## 7. Outer integral to construct

The target free energy per area is

```text
F/A = k_B T * sum_n' integral[d^2 Q / (2 pi)^2] L_n(Q, theta, d)
```

where `L_n` is the signed-real logdet returned from two compatible lab-basis reflections. The `n=0` prime weight is exactly `1/2` and belongs to the Matsubara quadrature layer, not to the microscopic response evaluator.

The first implementation should support:

- plate separation `d`;
- relative crystal angle `theta`;
- `spm` and `dwave` pairing;
- exact `n=0` plus positive Matsubara sum;
- q magnitude and q direction integration or a fully documented Cartesian alternative;
- free energy and torque;
- restartable cache and atomic outputs;
- convergence and sensitivity variants.

## 8. Recommended architecture

Keep the new runtime thin. Suggested layers:

1. `OuterIntegrationConfig`: temperature, separation, angles, pairing, q quadrature, Matsubara cutoff and tolerances.
2. `MicroscopicResponseProvider`: calls the retained total-Matsubara complete-orbit backend and returns validated reflection operators or signed logdet points.
3. `QQuadrature`: owns q magnitude/direction nodes and weights only.
4. `MatsubaraQuadrature`: owns prime weight, cutoff and tail diagnostics only.
5. `OuterAccumulator`: deterministic weighted summation, preferably compensated.
6. `SensitivityVariants`: exact-diagonal treatment and independent outer-grid variants.
7. `ReportWriter`: atomic metadata, compact summaries and local raw artifacts.

Do not place microscopic formulas in the CLI or outer quadrature classes.

## 9. Coordinate and basis rules

- The outer q variable is a lab-frame in-plane wavevector.
- Each plate receives `q_crystal = R(-theta_plate) q_lab` through the central basis convention.
- Both final reflection matrices must be represented in the same lab LT tangential-electric basis before multiplication.
- Do not revive old TE/TM-amplitude adapters or use the legacy complex trace-log diagnostic.
- Preserve model-q to SI-q conversion metadata at every point.
- Plate 1 and plate 2 reflections must have matching q, Matsubara frequency and vacuum kappa.

## 10. Exact-diagonal sensitivity variants

The first full trial must produce at least three variants:

### A. Native

Use the normal inner result at every outer node.

### B. Shifted-cut diagonal

At exact d-wave `qx=qy` nodes, use an independently shifted microscopic periodic cut.

### C. Symmetric observable interpolation

At exact d-wave diagonal nodes, replace only the final logdet by a symmetric interpolation from neighboring directions at the same q magnitude when the outer grid supports it.

Do not interpolate primitive response tensors unless a separate physical justification is established.

Compare pairwise free energy and torque differences. If the outer angular quadrature avoids exact diagonal nodes, retain a deliberate node-shift or alternative quadrature variant as the sensitivity comparison.

## 11. Convergence program

The minimum diagnostic convergence matrix is:

- q radial order/cutoff;
- q angular order or Cartesian grid spacing;
- Matsubara cutoff;
- angle grid used for torque;
- microscopic stage at representative high-weight points;
- native/shifted/interpolated exact-diagonal treatment;
- serial/process deterministic equality on a small outer grid.

Report both absolute and relative differences. Torque comparisons need a documented floor because torque vanishes at symmetry angles.

## 12. Suggested acceptance gates

For a diagnostic outer result:

```text
all pointwise physical pipelines passed = True
all signed-real logdet constructions passed = True
outer q-grid convergence <= target tolerance
Matsubara truncation/tail <= target tolerance
angle/torque discretization <= target tolerance
exact-diagonal variant sensitivity <= target tolerance
```

A reasonable initial target is `1e-3`, but the final tolerance must be stated explicitly in the outer integration config and reports.

Only after the complete convergence report may the project consider changing:

```text
production_reference_established = True
valid_for_casimir_input = True
```

No individual local-response threshold may change those flags by itself.

## 13. Forbidden shortcuts

Do not:

- construct zero frequency by dividing by Matsubara frequency;
- use bare or phase-only electromagnetic kernels;
- use local-q0 response as a finite-q fallback;
- silently take `real(logdet)` or `log(abs(det))`;
- mix reflection bases between plates;
- apply unvalidated q-direction symmetry reduction inside the microscopic integral;
- let diagnostic adaptive/profile routes become the production backend;
- commit raw outer grids, caches, full CSV tables or figures before a production reference exists;
- restore the retired monolithic Casimir script from Git history.

## 14. Output policy

Before production reference establishment, Git may retain only:

- `README.md`;
- exact reproduction command;
- run config;
- compact `summary` and `status` artifacts;
- handoff/convergence reports.

Raw arrays, full grids, caches, figures and logs remain local and are ignored by `.gitignore`.

## 15. First actions in the new window

1. Read this document and `validation/README.md`.
2. Confirm branch and head with `git status -sb` and `git rev-parse HEAD`.
3. Run the full test suite.
4. Regenerate a real-`nk` `orbit-gauss-preflight` manifest on the current head.
5. Design the outer config/schema and a tiny synthetic/small-grid end-to-end test before any expensive run.
6. Implement energy first; implement torque only after energy quadrature and angle conventions are tested.
7. Run native/shifted/interpolated diagonal sensitivity on a small full grid.
8. Expand q and Matsubara convergence only after atomic restart and reporting work.

## 16. Final handoff status

```text
microscopic two-band contract = ready for diagnostic outer construction
exact n=0 contract = ready
positive Matsubara contract = ready
signed-real single-point Lifshitz observable = ready
exact d-wave diagonal response reference = unresolved
exact d-wave diagonal observable sensitivity = locally below 1e-3
full q/angle/Matsubara convergence = not yet built
energy/torque production reference = not established

diagnostic_only = True
production_reference_established = False
valid_for_casimir_input = False
```
