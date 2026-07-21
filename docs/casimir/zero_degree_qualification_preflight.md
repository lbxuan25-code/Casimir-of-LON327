# SPM / d-wave 0° qualification rerun preflight

Status: **preflight frozen, execution not yet authorized**  
Target profile: `0deg_qualification_v5`  
Source evidence: the existing immutable `0deg_pilot_v4` runs  
Production authorization: **false until all gates in this report pass**

## 1. Purpose

This report freezes the numerical strategy for the next 0° qualification rerun of the SPM and d-wave pairings. The two pairings must use the same numerical structure, acceptance gates, adaptive ladders and error allocations. Pairing identity may change only the material response values, not the numerical decision rules.

The rerun must reuse the existing v4 certified-point caches. It must not start from an empty cache and must not modify either v4 source run.

Source runs:

```text
outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_pilot_v4
outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_pilot_v4
```

Target runs:

```text
outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_qualification_v5
outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_qualification_v5
```

## 2. Frozen physical case

Both runs use:

```text
temperature_K       = 10.0
separation_nm       = 20.0
plate_angles_deg    = (0.0, 0.0)
delta0_eV           = 0.1
eta_eV              = 1e-8
degeneracy          = 1.0
```

No physical model parameter is changed by the numerical qualification.

## 3. Frozen microscopic certification policy

The selected candidate is the least-relaxed stored-history candidate that closes the known d-wave boundary points in the completed convergence audit:

```text
logdet_rtol                  = 2.0e-3
logdet_atol                  = 1.0e-6
required_consecutive_passes  = 2
```

The following remain unchanged:

- all hard-physical response, Ward, sheet, passivity and determinant gates;
- cross-shift, adjacent-N and three-level oscillatory-envelope structure;
- the requirement for two consecutive accepted transitions;
- fail-closed handling of malformed, missing or non-finite evidence.

The global candidate comparison is therefore fixed as:

```text
strict reference  = 1.5e-3
selected moderate = 2.0e-3
aggressive audit  = 2.5e-3, not selected
stress boundary   = 3.0e-3, not selected
```

`2.5e-3` and `3.0e-3` must not be substituted after the run starts. A failure of the selected candidate triggers a new audit and a new holdout set rather than retuning against the same holdout evidence.

### Unified N ladder

Both pairings use the union ladder already represented by the v4 evidence:

```text
N_candidates = (
    128, 192, 256, 384, 512, 640, 768, 896,
    1024, 1152, 1280,
)
```

A point may stop earlier when the same gates pass. The code may not assign pairing-specific ladders.

## 4. Frozen finite-domain and total-energy policy

Both pairings use the same controller and budget split:

```text
radial_budget_fraction   = 0.80
angular_budget_fraction  = 0.20

total_free_energy_rtol        = 5.0e-3
total_free_energy_atol_J_m2   = 1.0e-12
```

The current pairing-dependent production default (`0.85` for SPM and `0.75` otherwise) is not allowed for this rerun. The workflow must expose and persist one common value before launch.

The finite-domain ledger must use the already-combined radial/angular/offset bound. Radial, angular and offset diagnostic components must not be added a second time.

## 5. Frozen outer-Q policy

Both pairings use the same cutoff ladder and geometric settings:

```text
cutoff_u_values = (6, 10, 14, 18, 24, 30, 36, 42, 48, 54, 60)
tail_start_u = 24
tail_window_shells = 3
tail_ratio_max = 0.8
```

The unified tail certificate has two possible paths:

1. **Resolved geometric path**: the central shell signal is numerically resolved and the final window contracts under the common ratio bound.
2. **Analytic passive-vacuum path**: a determinant-preserving power-metric certificate proves that the round-trip reflection operator is contractive, including singular-value rather than spectral-radius control, after which the pairing-independent analytic tail integral may be used.

Rules:

- `below_finite_domain_resolution` is diagnostic evidence only and never certifies the omitted tail by itself;
- a path participates only when all of its premises are certified;
- if both paths are valid, the final tail bound is the smaller valid bound;
- if neither path is valid, the run remains unresolved;
- SPM and d-wave use exactly the same logic.

The current repository only derives the analytic formula conditionally. The power-metric contraction certificate is therefore a launch blocker for any run that is expected to rely on the analytic path.

## 6. Frozen Matsubara policy

Both pairings use:

```text
matsubara_cutoff_values = (1, 3, 7, 11, 15, 23, 31)
matsubara_tail_start_n = 8
matsubara_tail_window_terms = 4
matsubara_tail_ratio_max = 0.8
```

No pairing-specific Matsubara ladder or tail rule is permitted.

## 7. Cache reuse contract

### 7.1 Source immutability

Before and after every preparation, holdout and production step, record and compare SHA-256 for:

```text
<source-v4>/config.json
<source-v4>/manifest.json
<source-v4>/result.json
<source-v4>/summary.json
<source-v4>/cache/certified_points.json
```

Any source change aborts the workflow.

### 7.2 Why the existing extension command is insufficient

The current `prepare-pilot-extension` path permits only an unchanged microscopic point policy or a strict N-ladder prefix extension. The v5 policy changes `logdet_rtol` from `1.5e-3` to `2.0e-3`, so using that command directly must fail rather than silently copying an incompatible cache.

A dedicated, audited policy-projection step is required.

### 7.3 Required policy-projection behavior

For every v4 cache entry:

1. retain the complete stored N/shift history;
2. replay the frozen v5 gates over that history with `logdet_rtol=2.0e-3`;
3. preserve every hard-physical result unchanged;
4. write a new v5 `sweet_spot` only when the frozen candidate establishes from stored evidence;
5. omit only entries that remain unresolved under the frozen v5 policy, so only those identities can trigger new microscopic work;
6. write the v5 point-policy payload and fingerprint;
7. record source/target SHA-256, retained/projected/omitted counts, every omitted identity and every changed decision in a projection report;
8. never overwrite an existing target cache without exact-policy validation.

This is reuse by verified history projection, not blind byte copying.

### 7.4 Expected reuse

- SPM v4 has established microscopic entries and should seed essentially all previously requested points.
- d-wave v4 should seed all strict-established entries and reclassify the four known boundary histories under the frozen moderate policy when the stored replay supports establishment.
- New microscopic work is allowed only for genuinely missing q/n identities, holdout levels, or points omitted because the frozen policy still does not establish them.
- Expanding the common outer cutoff to `u=60` may require new d-wave q points. Those are incremental additions, not a restart.

## 8. Independent high-N holdout

The candidate policy is frozen before the holdout is executed. The holdout set must include:

- all four known d-wave boundary points;
- the largest quadrature-weighted uncertainty contributors for each pairing and Matsubara channel;
- representative radii and directions;
- points that stop substantially earlier under `2.0e-3`;
- ordinary easy points as controls.

Use the existing weighted holdout plan as the starting list, with at most 32 primary points unless the required strata force a larger set. Each selected point is evaluated at the two predeclared N levels above its highest stored valid N.

Acceptance rule:

```text
abs(L_holdout - L_accepted) <= 2.0 * predicted_local_uncertainty
```

for every selected point and tested holdout level, with all hard-physical gates passing. The factor `2.0` is frozen before execution.

Holdout values are not used to retune `logdet_rtol`, the N ladder, or the safety factor. Failure rejects the candidate and requires a new candidate plus a new independent holdout.

## 9. Real-work benchmark

The holdout and incremental missing-point work must record:

- wall time;
- peak process memory where available;
- certifier batch count;
- new versus cache-hit q evaluations;
- new versus cache-hit point evaluations;
- cache byte growth;
- level-resolved certifier time.

Cache-only replay time and the N-squared work proxy remain diagnostics and do not count as the real-work benchmark.

## 10. Pre-launch code gates

The v5 run must not start until all of the following are implemented and tested:

1. remove the pairing-dependent radial-budget default or add a required common `radial_budget_fraction` input persisted in `config.json`;
2. add the audited v4-to-v5 policy-projection cache command described above;
3. add a holdout executor that writes immutable requested identities/N levels before computing values;
4. add and validate the power-metric contraction certificate if the analytic tail path will be allowed to pass production acceptance;
5. make the production tail controller choose only among premise-valid bounds and apply identical logic to both pairings;
6. emit a preflight manifest containing the frozen policy, source hashes, target paths, holdout-plan hash and Git commit;
7. test that both pairings produce identical numerical-policy snapshots after physical identity is excluded.

Until these gates pass, the status remains:

```text
ready_to_prepare_code = true
ready_to_seed_v5_cache = false
ready_to_run_v5 = false
production_change_authorized = false
```

## 11. Execution order after the code gates pass

1. freeze the Git commit and write the preflight manifest;
2. hash both v4 source runs;
3. create the two v5 caches by audited policy projection;
4. verify target fingerprints, projection reports and source immutability;
5. execute the independent holdout and real-work benchmark;
6. if the holdout passes, run SPM and d-wave 0° with the same complete CLI policy;
7. regenerate diagnostics and the unified convergence audit;
8. verify microscopic, finite-domain, outer-tail, Matsubara-tail and total-energy ledgers;
9. refresh the run catalog and register v5 only after artifacts are complete;
10. keep v4 as frozen evidence until v5 is formally accepted.

## 12. Final acceptance contract

Each pairing must independently satisfy:

```text
all_microscopic_points_established = true
independent_high_N_holdout_passed = true
finite_domain_budget_passed = true
outer_tail_bound_certified = true
matsubara_tail_bound_certified = true
nonduplicating_total_error_within_tolerance = true
source_v4_cache_unchanged = true
```

The joint decision additionally requires:

```text
pairing_blind_numerical_policy = true
same_tail_controller_structure = true
same_error_allocation = true
real_work_benchmark_recorded = true
```

A failure in one pairing does not authorize a pairing-specific exception. It leaves that pairing unresolved and triggers a new common-policy audit.

## 13. Frozen decision

The next qualification candidate is therefore:

```text
profile                       = 0deg_qualification_v5
logdet_rtol                   = 2.0e-3
logdet_atol                   = 1.0e-6
required_consecutive_passes   = 2
N_candidates                  = 128..1280 unified ladder
radial/angular budget         = 0.80 / 0.20
outer cutoff ladder           = 6..60 unified ladder
Matsubara ladder              = 1,3,7,11,15,23,31
hard physical gates           = unchanged
source cache                  = each pairing's immutable 0deg_pilot_v4 cache
cache strategy                = audited history projection, never empty restart
```

This is a frozen qualification policy, not yet a production authorization.
